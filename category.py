import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import torch
import numpy as np
import cv2
from utils.helpers import maybe_download
from models import rf_lw50, rf_lw101, rf_lw152, rf_lw50c, rf_lw101c, rf_lw152c, rf_lw50c_i, rf_lw101c_i, rf_lw152c_i


# load image
def load_npy_file(file_path):
    return np.load(file_path)

# process image
def process_image(image,input_size):
    image = cv2.resize(image, (input_size, input_size))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image

# psotprocess image
def postprocess_image(image, output_size):
    image=  output_size.squeeze(0) #remove batch dimension
    image=  output_size.argmax(0) #get the class index with the highest probability
    image= image.cpu().numpy()  #convert to numpy array
    image= cv2.resize(output_size,output_size, interpolation=cv2.INTER_NEAREST) #resize to original size
    return image

# segment image
def segment_image(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        output = postprocess_image(output, (image.shape[2], image.shape[3]))

    return output
# process each file
def process_file(file_path,model,input_size):
    #load image
    image = load_npy_file(file_path)
    #process image
    image = process_image(image, input_size)
    imal =torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0) #clear batch dimension
    #segment image
    output = segment_image(image, model)
    #save and display image
    output_file_path = os.path.splitext(file_path)[0] + '_output.png'
    cv2.imwrite(output_file_path, output)
    print('Segmentation done and saved as {}'.format(output_file_path))


# main function
def main(npy_file_path,model_type='rf_1w50',num_classes=21,input_size=512):
  
    if model_type == 'rf_1w50':
        model = rf_lw50(num_classes)
    elif model_type == 'rf_1w101':
        model = rf_lw101(num_classes)
    elif model_type == 'rf_1w152':
        model = rf_lw152(num_classes)
    elif model_type == 'rf_1w50c':
        model = rf_lw50c(num_classes)
    elif model_type == 'rf_1w101c':
        model = rf_lw101c(num_classes)
    elif model_type == 'rf_1w152c':
        model = rf_lw152c(num_classes)
    elif model_type == 'rf_1w50c_i':
        model = rf_lw50c_i(num_classes)
    elif model_type == 'rf_1w101c_i':
        model = rf_lw101c_i(num_classes)
    elif model_type == 'rf_1w152c_i':
        model = rf_lw152c_i(num_classes)
    else:
        raise ValueError('Invalid model type')
    
    #process all file
    for root, dirs, files in os.walk(npy_file_path):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path,model,input_size)

if __name__ == '__main__':
    npy_file_path = "/home/yanwenli/light-weight-refinenet/test/data"
    main(npy_file_path)