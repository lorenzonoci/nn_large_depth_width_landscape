import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.transforms import ToTensor
import os
from PIL import Image

# Load CIFAR10 training dataset
train_data = CIFAR10(root='/is/cluster/fast/ameterez/datasets/', train=True, download=True, transform=ToTensor())

# Extract data and labels for stratification
X = torch.tensor(train_data.data)
y = torch.tensor(train_data.targets)

# Stratified subsampling of 5000 samples from the train set
sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=0)
for _, subsample_index in sss.split(X, y):
    subsampled_indices = subsample_index

# Create a subset of the original training dataset
subsampled_dataset = Subset(train_data, subsampled_indices)

# Save subsampled dataset in a directory
save_dir = './subsampled_cifar10_train'
os.makedirs(save_dir, exist_ok=True)

# Function to save images from the Subset
def save_images(dataset, save_dir):
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image = (image * 255).byte()  # Convert back to PIL image format
        img = Image.fromarray(image.numpy().transpose((1, 2, 0)), 'RGB')
        
        # Define path based on label for organized saving
        label_dir = os.path.join(save_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f'{idx}.png')
        
        img.save(img_path)

# Save the subsampled images
save_images(subsampled_dataset, save_dir)
