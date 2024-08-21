import numpy as np
from scipy import io

def load_npy_file(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    return data

def get_image_dimensions(image):
    height, width, _ = image['rgb'].shape
    return height, width

def get_label_dimensions(image):
    labe_image = data.get('label')
    return labe_image.shape

def visualize_image(image):
    import matplotlib.pyplot as plt
    rgb_image = image.get('rgb')
    plt.imshow(rgb_image)
    plt.show()

if __name__ == "__main__":
    npy_file_path = "/home/yanwenli/light-weight-refinenet/test/data/1/scene.npy"
    image = load_npy_file(npy_file_path)
    print(type(image))
    print(image)
    height, width = get_image_dimensions(image)
    print(f"Height: {height}, Width: {width}")
    label_image = image.get('label')    
    print(f"Label image shape: {label_image.shape}")
    visualize_image(image)

# Load the .npy file
npy_file_path = '/home/yanwenli/light-weight-refinenet/test/data/1/scene.npy'
mat = np.load(npy_file_path, allow_pickle=True).item()

# Save the dictionary to a .mat file
mat_file_path = '/home/yanwenli/light-weight-refinenet/test/data/1/scene.mat'
io.savemat(mat_file_path, mat)

print(f"Successfully saved {npy_file_path} as {mat_file_path}")