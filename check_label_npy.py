import numpy as np

# Load the .npy file
file_path = "/home/yanwenli/light-weight-refinenet/test/data/1/scene.npy"
data = np.load(file_path, allow_pickle=True).item()

# Make sure the data contains the 'label' key
if 'label' in data:
    label_image = data['label']
    print("Label image shape:", label_image.shape)

    # Find all unique classes in the label image
    unique_labels = np.unique(label_image)
    print("Unique labels in the image:", unique_labels)

    # Output the number of classes
    num_classes = len(unique_labels)
    print("Number of classes:", num_classes)
    
    # Check the pixel count for each class
    for label in unique_labels:
        count = np.sum(label_image == label)
        print(f"Number of pixels for label {label}: {count}")
else:
    print("No 'label' key found in the loaded data")

