import torch.utils.data as data
import os
from PIL import Image
import random
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from torch.utils.data import DataLoader

class Haze1kDataset(data.Dataset):
    def __init__(self, path, train, size, format='.png'):
        super(Haze1kDataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_dir = os.path.join(path, 'input')
        self.haze_imgs_name = sorted(
            os.listdir(self.hazy_dir),
            key=lambda x: int(x.split('-')[0])
        )
        self.haze_imgs_dir = [os.path.join(self.hazy_dir, img) for img in self.haze_imgs_name]
        self.clear_dir = os.path.join(path, 'target')

    def __getitem__(self, index):
        haze_image = Image.open(self.haze_imgs_dir[index])
        if isinstance(self.size, int):
            while haze_image.size[0] < self.size or haze_image.size[1] < self.size:
                index = random.randint(0, 20000)
                haze_image = Image.open(self.haze_imgs_dir[index])

        haze_img_dir = self.haze_imgs_dir[index]
        id = haze_img_dir.split('/')[-1].split('-')[0]
        id_num = int(id)
        clear_img_name = id + '-targets' + self.format
        clear_image = Image.open(os.path.join(self.clear_dir, clear_img_name))
        clear_image = tfs.CenterCrop(haze_image.size[::-1])(clear_image)
        if self.train and isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze_image, output_size=(self.size, self.size))
            haze_image = FF.crop(haze_image, i, j, h, w)
            clear_image = FF.crop(clear_image, i, j, h, w)
        haze, clear = self.augData(haze_image.convert("RGB"), clear_image.convert("RGB"))
        return haze, clear, id_num

    def augData(self, haze, clear):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                haze = FF.rotate(haze, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs_dir)


class RRSHIDDataset(data.Dataset):
    def __init__(self, path, train, size=(512,512), format='.png'):
        super(RRSHIDDataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_dir = os.path.join(path, 'hazy')
        self.haze_imgs_name = sorted(
            os.listdir(self.hazy_dir),
            key=lambda x: int(x.split('.')[0])
        )
        self.haze_imgs_dir = [os.path.join(self.hazy_dir, img) for img in self.haze_imgs_name]
        self.clear_dir = os.path.join(path, 'gt')

    def __getitem__(self, index):
        haze_image = Image.open(self.haze_imgs_dir[index])
        if isinstance(self.size, int):
            while haze_image.size[0] < self.size[0] or haze_image.size[1] < self.size[1]:
                index = random.randint(0, 20000)
                haze_image = Image.open(self.haze_imgs_dir[index])

        haze_img_dir = self.haze_imgs_dir[index]
        id = haze_img_dir.split('/')[-1].split('.')[0]
        clear_img_name = id + self.format
        clear_image = Image.open(os.path.join(self.clear_dir, clear_img_name))
        clear_image = tfs.CenterCrop(haze_image.size[::-1])(clear_image)
        if self.train and isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze_image, output_size=(self.size[0], self.size[1]))
            haze_image = FF.crop(haze_image, i, j, h, w)
            clear_image = FF.crop(clear_image, i, j, h, w)
        haze, clear = self.augData(haze_image.convert("RGB"), clear_image.convert("RGB"))
        return haze, clear

    def augData(self, haze, clear):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                haze = FF.rotate(haze, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs_dir)


class DHIDDataset(data.Dataset):
    def __init__(self, path, train, size=(512,512), format='.jpg'):
        super(DHIDDataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_dir = os.path.join(path, 'hazy')
        self.haze_imgs_name = sorted(
            os.listdir(self.hazy_dir),
            key=lambda x: (
                int(x.split('.')[0].split('_')[0]),
                int(x.split('.')[0].split('_')[1]),
            )
        )
        self.haze_imgs_dir = [os.path.join(self.hazy_dir, img) for img in self.haze_imgs_name]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze_image = Image.open(self.haze_imgs_dir[index])
        if isinstance(self.size, int):
            while haze_image.size[0] < self.size[0] or haze_image.size[1] < self.size[1]:
                index = random.randint(0, 20000)
                haze_image = Image.open(self.haze_imgs_dir[index])

        haze_img_dir = self.haze_imgs_dir[index]
        id = haze_img_dir.split('/')[-1].split('_')[0]
        clear_img_name = id + self.format
        clear_image = Image.open(os.path.join(self.clear_dir, clear_img_name))
        clear_image = tfs.CenterCrop(haze_image.size[::-1])(clear_image)
        if self.train and isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze_image, output_size=(self.size[0], self.size[1]))
            haze_image = FF.crop(haze_image, i, j, h, w)
            clear_image = FF.crop(clear_image, i, j, h, w)
        haze, clear = self.augData(haze_image.convert("RGB"), clear_image.convert("RGB"))
        return haze, clear

    def augData(self, haze, clear):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                haze = FF.rotate(haze, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs_dir)


class LHIDDataset(data.Dataset):
    def __init__(self, path, train, size=(512,512), format='.jpg'):
        super(LHIDDataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_dir = os.path.join(path, 'hazy')
        self.haze_imgs_name = sorted(
            os.listdir(self.hazy_dir),
            key=lambda x: int(x.split('_')[0])
        )
        self.haze_imgs_dir = [os.path.join(self.hazy_dir, img) for img in self.haze_imgs_name]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze_image = Image.open(self.haze_imgs_dir[index])
        if isinstance(self.size, int):
            while haze_image.size[0] < self.size[0] or haze_image.size[1] < self.size[1]:
                index = random.randint(0, 20000)
                haze_image = Image.open(self.haze_imgs_dir[index])

        haze_img_dir = self.haze_imgs_dir[index]
        id = haze_img_dir.split('/')[-1].split('_')[0]
        clear_img_name = id + self.format
        clear_image = Image.open(os.path.join(self.clear_dir, clear_img_name))
        clear_image = tfs.CenterCrop(haze_image.size[::-1])(clear_image)
        if self.train and isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze_image, output_size=(self.size[0], self.size[1]))
            haze_image = FF.crop(haze_image, i, j, h, w)
            clear_image = FF.crop(clear_image, i, j, h, w)
        haze, clear = self.augData(haze_image.convert("RGB"), clear_image.convert("RGB"))
        return haze, clear

    def augData(self, haze, clear):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                haze = FF.rotate(haze, 90 * rand_rot)
                clear = FF.rotate(clear, 90 * rand_rot)

        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        return haze, clear

    def __len__(self):
        return len(self.haze_imgs_dir)






