import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from randaugment import RandomAugment

# CLIP image normalization constants (mean and std over ImageNet)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transform(args):
    """Build the training image transform pipeline."""
    normalize = transforms.Normalize(_CLIP_MEAN, _CLIP_STD)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.image_res, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(
            2, 7, isPIL=True,
            augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'],
        ),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

class ImgSentDataset(Dataset):
    def __init__(self,
                 text_file,
                 feature_file=None,
                 image_root=None,
                 transform=None,
                 shuffle_imgs=False,
                 random_imgs=False,
                 shot=-1):

        self.text_file = text_file
        self.feature_file = feature_file
        self.image_root = image_root
        self.transform = transform
        self.shuffle_imgs = shuffle_imgs
        self.random_imgs = random_imgs
        self.shot = shot
        self.raw_dataset = self.load_data()

    def load_data(self):
        data = []
        sentonly = True if self.feature_file is None else False

        # loading sentences
        with open(self.text_file) as f:
            sentences = [l.strip() for l in f.readlines()]

        N = len(sentences)

        # loading image features
        if not sentonly:
            with open(self.feature_file) as f:
                clip_data = json.load(f)

            for k in tqdm(clip_data, desc="Loading image-text pairs"):
                img = torch.tensor(clip_data[k]['image_feat'])
                image_path = os.path.join(self.image_root, clip_data[k]['image'])
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)  # (C, H, W)
                for ic in range(len(clip_data[k]['captions'])):
                    sent = clip_data[k]['captions'][ic]
                    clip_feat = torch.tensor(clip_data[k]['lang_feat'][ic])
                    d = {'image': image, 'sent': sent, 'img': img,
                         'clip_text_feat': clip_feat, 'img_key': k}
                    data.append(d)

        else:
            for sent in sentences:
                d = {'sent': sent}
                data.append(d)

        if self.shot > 0:
            index = np.random.choice(N, self.shot, replace=False)
            data = np.array(data)[index].tolist()

        return data


    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item:int):
        datum = self.raw_dataset[item]

        return datum



