"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import torch
import numpy as np
from warnings import warn

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset_with_attrs(dir, attr_dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    attrs = np.load(attr_dir) 
    # attrs[attrs == -1] = 0 ## one-hot encoding had not been tested
    # b = np.zeros((attrs.size, attrs.max() + 1))
    # b[np.arange(attrs.size), attrs] = 1
    # attrs = b.astype(int)
    attrs[attrs==-1] = 0
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                attr = attrs[int(fname.replace('.jpg', ''))]
                images.append((path, attr))

            elif fname.endswith('.npy'):
                path = os.path.join(root, fname)
                attr = attrs[int(fname.replace('.jpg', ''))]
                images.append((path, attr))


    return images



class ImageFolder_with_attributes(data.Dataset):

    def __init__(self, root, transform=None, attr_path=None, return_paths=False,
                 loader=default_loader):
        
        imgs = sorted(make_dataset_with_attrs(root, attr_path))
        
        # if attr_path == './datasets/Bitmoji2Face/bitmoji_genders.npy':
        #     print(imgs[:10])
        #     exit()
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

        # TODO tmp
        self.mean = 0
        self.std = None
        self.mean_pow2 = 0

        self.num_of_sample = 0


    def __getitem__(self, index):
        index = index % len(self.imgs)
        path, attr = self.imgs[index]
        if not path.endswith('.npy'):
            try:
                img = self.loader(path)
                if self.transform is not None:
                    img = self.transform(img)
            except Exception as e:
                print(str(e))
                warn(f'Failed to load {path}, removing from images list')
                del self.imgs[index]
                index = index % len(self.imgs)
                if self.return_paths:
                    img, path = self.__getitem__(self, index)
                else:
                    img = self.__getitem__(self, index)

        else:  # numpy data input # for brats dataset  # TODO add transforms
            img = torch.from_numpy(np.load(path))
            img = img.transpose(dim0=2, dim1=0)
            img = img.transpose(dim0=1, dim1=2)
            img = img.to(dtype=torch.float)
            from torchvision import transforms
            import torch.nn.functional as F
            mean_vec = torch.mean(img, dim=(1, 2))
            std_vec = torch.std(img, dim=(1, 2))
            img = transforms.Normalize(mean=mean_vec, std=std_vec)(img)
            size = [elem for elem in self.transform.transforms if type(elem) == transforms.transforms.Resize][0].size
            img = F.interpolate(input=torch.unsqueeze(img,0), size=size)
            img = torch.squeeze(img)
            # img = img[:-1,:,:]

            # dim0 Flair
            # dim1 T1
            # dim2 T1ce
            # dim3 T2
            # dim4 Seg

        if self.return_paths:
            return img, path
        else:
            return img, attr

    def __len__(self):
        return len(self.imgs)

