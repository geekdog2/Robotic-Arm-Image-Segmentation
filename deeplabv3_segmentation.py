import numpy as np
import cv2
import torch
import os
from torchvision import models

def get_color_map(num_classes):
    """Generate a color map with a unique color for each class."""
    np.random.seed(42)
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    return colors

def load_npy_image(file_path):
    """Load .npy file and return the image data."""
    data = np.load(file_path, allow_pickle=True).item()
    if isinstance(data, dict) and 'rgb' in data:
        return data['rgb']
    else:
        raise ValueError("Loaded data is not in the expected format or 'rgb' key is missing")

def process_image(image, input_size=256):
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

def preprocess_and_infer(image, model, device, input_size=256):
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

def load_model(device, num_classes=30):
    """加载 DeepLabV3 模型并转移到指定设备。"""
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)  # 使用预训练的 DeepLabV3 模型
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))  # 修改分类器以适应新的类别数
    model.to(device)
    return model

def process_npy_file(input_dir, output_dir, device, model, input_size=256):
    """处理每个 .npy 文件，执行图像分割，并保存结果。"""
    counter = 1  # Initialize counter
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    color_map = get_color_map(num_classes=30)  # set the number of classes
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                try:
                    # Load and process image
                    image = load_npy_image(file_path)
                    output = preprocess_and_infer(image, model, device, input_size)
                    
                    # Get segmentation results
                    segm = torch.argmax(output, dim=1).cpu().numpy().squeeze()
                    
                    # 将灰度分割结果转换为彩色图像
                    segm_color = color_map[segm]
                    
                    # Save segmentation result
                    base_name = os.path.splitext(file)[0]
                    output_file_path = os.path.join(output_dir, f"{base_name}_segmented_{counter}.png")
                    cv2.imwrite(output_file_path, segm_color)
                    print(f"Segmentation done and saved as {output_file_path}")
                
                    counter += 1  # Update counter
                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")

if __name__ == "__main__":
    input_dir = "/home/yanwenli/light-weight-refinenet/test/data"
    output_dir = "/home/yanwenli/light-weight-refinenet/test/output_images"
    input_size = 256
      # 可以根据你的模型修改输入尺寸
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(device, num_classes=30) 
    process_npy_file(input_dir, output_dir, device, model, input_size=input_size)
