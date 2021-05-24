import os
import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class GolfDataset(Dataset):
    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.images = sorted(os.listdir(os.path.join(root, "image")))
        self.labels = sorted(os.listdir(os.path.join(root, "label/bb")))

    def __getitem__(self, index):
        image_path = os.path.join(self.root, 'images', self.images[index])
        lable_path = os.path.join(self.root, 'label/bb', self.labels[index])
        img = Image.open(image_path).convert('RGB')
        img = np.asarray(img)

        f = open(lable_path)
        boxes = lable_path[lable_path.split(
            '_')[-1].split('.')[0] + '_detect.json'][1]['bb']['person_bb']
        target = {}
        target['boxes'] = boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = GolfDataset(root='/media/tk/SSD_250G')
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True)

    for epoch in range(1):
        for i, data in enumerate(data_loader):
            print(data)
