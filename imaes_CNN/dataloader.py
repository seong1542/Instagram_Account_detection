import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
import tqdm
import glob
import cv2
from cv2 import resize

train_transforms = transforms.Compose([transforms.Resize((160,350)),
                                       transforms.RandomResizedCrop(160),
                                       transforms.ToTensor(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Normalize([0.155,0.5,0.885], [0.229,0.224,0.225])])

class CustomDataset(Dataset):
    def __init__(self, mode='train', width=320, height=700):
        self.mode = mode
        self.path = './data'
        self.width = int(width)
        self.height = int(height)


        # train / test 분리 시 활용
        if self.mode == 'train':
            self.path += '/train/'
        elif self.mode == 'val':
            self.path += '/val/'
        else:
            self.path += '/test/'


        self.real_files = sorted(glob.glob(self.path + 'Real/*.jpg'))
        self.fake_files = sorted(glob.glob(self.path + 'Fake/*.jpg'))
        self.transforms=transforms

        self.x_data, self.y_data = [], []
        for real_file in tqdm.tqdm(self.real_files):
            img = np.array(resize(cv2.imread(real_file), dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST))
            self.x_data.append(img)
            self.y_data.append(0) # Real

        for fake_file in tqdm.tqdm(self.fake_files):

            img = np.array(resize(cv2.imread(fake_file), dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST))
            self.x_data.append(img)
            self.y_data.append(1) # Fake



        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data).reshape(-1, 1)
        print(self.x_data.shape, self.y_data.shape)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.x_data[idx]).view(3, self.height, self.width), torch.LongTensor(self.y_data[idx])

        return sample


if __name__ == "__main__":
    train_dataset = CustomDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, transfrom=train_transforms)

    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        print(data.shape, target.shape)

        break
