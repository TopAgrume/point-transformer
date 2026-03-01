import os
import numpy as np
import torch
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create

class ModelNet40(Dataset):
    def __init__(self, split='train', data_root='dataset/modelnet40_normal_resampled', num_points=4000, transform=None, loop=1):
        super().__init__()
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.loop = loop

        # 1. Load shape names to map categories to labels (0-39)
        shape_names_path = os.path.join(data_root, 'modelnet40_shape_names.txt')
        with open(shape_names_path, 'r') as f:
            shape_names = [line.strip() for line in f.readlines()]
        self.classes = dict(zip(shape_names, range(len(shape_names))))

        # 2. Load the specific split list (train or test)
        split_list_path = os.path.join(data_root, f'modelnet40_{split}.txt')
        with open(split_list_path, 'r') as f:
            self.data_list = [line.strip() for line in f.readlines()]

        # 3. Cache data into SharedArray for fast, multi-worker memory access
        for item in self.data_list:
            if not os.path.exists(f"/dev/shm/{item}"):
                category = '_'.join(item.split('_')[0:-1])

                # ModelNet40 is commonly stored as comma-separated .txt files
                # (x, y, z, nx, ny, nz) but we also check for .npy to support pre-processed data
                data_path_txt = os.path.join(data_root, category, item + '.txt')
                data_path_npy = os.path.join(data_root, category, item + '.npy')

                if os.path.exists(data_path_npy):
                    data = np.load(data_path_npy)
                else:
                    data = np.loadtxt(data_path_txt, delimiter=',')

                sa_create(f"shm://{item}", data)

        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        item_name = self.data_list[data_idx]

        # Extract the global classification label from the filename
        category = '_'.join(item_name.split('_')[0:-1])
        label = self.classes[category]

        # Attach to the shared memory array
        data = SA.attach(f"shm://{item_name}").copy()

        # Separate coordinates and normal vectors
        coord, feat = data[:, 0:3], data[:, 3:6]

        # Uniformly sample points if the shape exceeds num_points
        if coord.shape[0] > self.num_points:
            choice = np.random.choice(coord.shape[0], self.num_points, replace=False)
            coord = coord[choice, :]
            feat = feat[choice, :]

        # Apply spatial augmentations (e.g., RandomScale, RandomTranslate)
        if self.transform is not None:
            coord, feat, label = self.transform(coord, feat, label)

        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        label = torch.LongTensor([label])

        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop