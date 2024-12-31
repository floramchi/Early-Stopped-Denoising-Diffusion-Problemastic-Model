import torch
import numpy as np
import torch.utils.data as data
from PIL import Image  # To resize the images
# import h5py
import pdb

class NpzDataset(data.Dataset):
    def __init__(self, data_dir, normalize=True, permute=True, rank=0, world_size=1, num_samples=1000, image_size=(128, 128)):
        # Load the .npz file
        data = np.load(data_dir)
        self.input_data = data['arr_0']  # B,H,W,3
        if 'arr_1' in data.files:
            self.labels = data['arr_1'].astype(int)
        else:
            self.labels = np.zeros(self.input_data.shape[0], dtype=int)
        data.close()

        # Limit to num_samples (5000 images)
        self.input_data = self.input_data[:num_samples]
        self.labels = self.labels[:num_samples]

        # Resize images to the specified image_size (128, 128)
        self.image_size = image_size
        self.input_data = np.array([self.resize_image(img) for img in self.input_data])

        if permute:  # input_data is of shape B,H,W,3
            self.input_data = self.input_data.transpose(0,3,1,2)  # Now it is of shape B,3,H,W

        if world_size > 1:
            num_samples_per_rank = int(np.ceil(self.input_data.shape[0] / world_size))
            start = rank * num_samples_per_rank
            end = (rank + 1) * num_samples_per_rank
            self.input_data = self.input_data[start:end]
            self.labels = self.labels[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = self.input_data.shape[0]

        if normalize:
            self.input_data = (self.input_data.astype(np.float32) / 255) * 2 - 1  # Normalize to [-1, 1]

        print('dataset %s:' % data_dir)
        print('input_data:', self.input_data.shape)
        print('labels:', self.labels.shape)
        self.len = self.input_data.shape[0]
        # print(f"Dataset size: {self.input_data.shape[0]} images, Image shape: {self.input_data.shape[2:]}")


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.input_data[index]
        label = self.labels[index]
        return x, label
    

    def resize_image(self, img):
        # Convert numpy array (H, W, C) to a PIL Image, then resize
        img = Image.fromarray(img)
        img = img.resize(self.image_size)  # Resize image to the target size
        return np.array(img)  # Convert back to numpy array

class DummyDataset(data.Dataset):
    def __init__(self, num_samples, rank=0, world_size=1):
        self.input_data = np.arange(num_samples)
        if world_size > 1:
            num_samples_per_rank = int(np.ceil(self.input_data.shape[0] / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.input_data = self.input_data[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = self.input_data.shape[0]

        print('dummy dataset:')
        print('input_data:', self.input_data.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.input_data[index]
        return x, 0

