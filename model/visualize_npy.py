import numpy as np
import matplotlib.pyplot as plt
import os


# Dynamically get the home directory
home_dir = os.path.expanduser("~")
# Construct the desired path relative to the home directory
npy_path = os.path.join(
    home_dir,
    "dev",
    "storage",
    "1",
    "scene.npy",
)

print(npy_path)

# Load the numpy array
data = np.load(npy_path, allow_pickle=True).item()

print(data)


# # extract the keys
rgb_image = data["rgb"]

plt.imshow(rgb_image)
plt.show()
# Display the image
# for i,img in enumerate(rgb_image):
#     plt.imshow(img)
#     plt.title(f"Image {i+1}")
#     plt.show()