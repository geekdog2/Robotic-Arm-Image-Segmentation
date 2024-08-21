import numpy as np
import cv2
import torch
import os
from models.mobilenet import mbv2


def load_npy_image(file_path):
    """加载 .npy 文件并返回图像数据。"""
    data = np.load(file_path, allow_pickle=True).item()
    if isinstance(data, dict) and 'rgb' in data:
        return data['rgb']
    else:
        raise ValueError("Loaded data is not in the expected format or 'rgb' key is missing")

def process_image(image, input_size=224):
    """预处理图像，例如调整大小、归一化等。"""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)  # 确保图像数据类型为 uint8
        
        # Resize image to the input size of the model
        image = cv2.resize(image, (input_size, input_size))
        
        # Convert to float and normalize the image
        image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1] 之间
        
        # Convert to (C, H, W) format
        image = np.transpose(image, (2, 0, 1))  # 从 HWC 转换为 CHW
    else:
        raise TypeError("Expected input image to be of type np.ndarray")
    
    return image

def preprocess_and_infer(image, model, device, input_size=224):
    """将图像数据预处理并传入模型进行推理。"""
    # 预处理图像
    image = process_image(image, input_size)
    
    # 将图像数据转换为 torch.Tensor 并调整维度
    image = torch.tensor(image).unsqueeze(0).to(device)  # (1, C, H, W)
    
    # 执行推理
    model.eval()
    with torch.no_grad():
        output = model(image)['out']
    
    return output

def load_model(device, num_classes=21):
    """加载轻量级 RefineNet 模型并转移到指定设备。"""
    model = mbv2(num_classes=num_classes, pretrained=False)  # 使用你提供的轻量级 RefineNet 模型
    model.eval()
    model.to(device)
    return model

def process_npy_file(input_dir, output_dir, device, model, input_size=224):
    """处理每个 .npy 文件，执行图像分割，并保存结果。"""
    counter = 1 #initialize counter
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                try:
                    # 加载和处理图像
                    image = load_npy_image(file_path)
                    image = process_image(image, input_size)
                    image = torch.tensor(image).float().unsqueeze(0).permute(0, 1, 2, 3)  # 转换为 (N, C, H, W) 格式
                    image = image.to(device)
                    
                    # 执行推理
                    with torch.no_grad():
                        output = model(image)
                    
                    # 获取预测结果
                    segm = torch.argmax(output, dim=1).cpu().numpy().squeeze()
                    
                     # 保存分割结果
                    base_name = os.path.splitext(file)[0]
                    output_file_path = os.path.join(output_dir, f"{base_name}_segmented_{counter}.png")
                    if segm.ndim == 2:  # 确保分割结果是二维图像
                        segm = (segm * 255 / np.max(segm)).astype(np.uint8)  # 归一化到 [0, 255] 之间
                        segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)  # 将灰度图像转换为BGR格式
                    cv2.imwrite(output_file_path, segm)
                    print(f"Segmentation done and saved as {output_file_path}")
                
                    counter += 1  # 更新计数器
                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")



if __name__ == "__main__":
    input_dir = "/home/yanwenli/light-weight-refinenet/test/data"
    output_dir = "/home/yanwenli/light-weight-refinenet/test/output_images"
    input_size = 224  # 可以根据你的模型修改输入尺寸
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(device, num_classes=30) 
    process_npy_file(input_dir, output_dir, device, model, input_size=input_size)

