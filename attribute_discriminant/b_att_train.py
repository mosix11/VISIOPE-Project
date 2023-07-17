import glob
import numpy as np
import torch
import torchvision
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from model import BitmojiGenderClassifier


train_data_dir = "../datasets/Bitmoji2Face/trainB/*"
train_attrs_dir = "../datasets/Bitmoji2Face/bitmoji_genders.npy"
model_weights_path = "./b_model_weights"
batch_size = 128
num_classes = 2
train_val_spilt = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_image(path):
    img = io.imread(path)
    img = transform.resize(img, (256, 256), anti_aliasing=True)
    return img

class MaDataset(Dataset):

    def __init__(self, img_paths, attributes, transform=None):
        self.attributes = attributes
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fpath = self.img_paths[idx]
        image = io.imread(fpath)
        attrs = self.attributes[idx]
        attrs = np.array([1, 0]) if attrs == 0 else np.array([0, 1])
        attrs = torch.tensor(attrs, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        return image.to(device), attrs.to(device)






@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history


  


files_path = glob.glob(train_data_dir)
files_path = sorted(files_path)
genders = np.load(train_attrs_dir)
genders[genders == -1] = 0

dataset_size = len(files_path)
train_size = math.floor(dataset_size*train_val_spilt)
test_size = dataset_size - train_size

train_paths = files_path[:train_size]
train_genders = genders[:train_size]

test_paths = files_path[train_size:]
test_genders = genders[train_size:]


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomGrayscale(0.05),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    
])


train_dataset = MaDataset(train_paths, train_genders, train_transforms)
test_dataset = MaDataset(test_paths, test_genders)

train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(train_dataset, batch_size, False)






num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001
model = BitmojiGenderClassifier().to(device)

# history = fit(num_epochs, lr, model, train_loader, test_loader, opt_func)

# torch.save(model.state_dict(), model_weights_path)

model.load_state_dict(torch.load(model_weights_path))
model.eval()


t = 0

for i in range(test_size):
    fpath = test_paths[i]
    image = load_image(fpath)
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).to(device)
    attrs = test_genders[i]
    attrs = np.array([1, 0]) if attrs == 0 else np.array([0, 1])
    attrs = torch.tensor(attrs, dtype=torch.float32).to(device)
    
    
    
    image = image.unsqueeze(0)
    attrs = attrs.unsqueeze(0)
    
    # print(image.shape)
    # print(attrs.shape)
    
    out_data = model(image)
    
    if attrs.tolist() == out_data.tolist():
        t += 1
    
    
    # imgs, labels = batch
    # out_data = model(imgs)
    # print(out_data)

print(t/test_size)

# for i, sample in enumerate(train_dataset):
#     print(i, sample[0].shape, sample[1].shape)
#     if i == 10:
#         break