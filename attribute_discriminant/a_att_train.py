import glob
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from model import BitmojiGenderClassifier


train_data_dir = "../datasets/Bitmoji2Face/trainA/*"
train_attrs_dir = "../datasets/Bitmoji2Face/celeba_genders.npy"
model_weights_path = "./a_model_weights"
batch_size = 128
num_classes = 2
train_val_spilt = 0.80

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_image(path):
    img = io.imread(path)
    img = transform.resize(img, (256, 256))
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


        return image, attrs






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
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomGrayscale(0.05),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
    
])


train_dataset = MaDataset(train_paths, train_genders, train_transforms)
test_dataset = MaDataset(test_paths, test_genders)

train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(train_dataset, batch_size, False)






# num_epochs = 30
# opt_func = torch.optim.Adam
# lr = 0.001
# model = BitmojiGenderClassifier().to(device)

# history = fit(num_epochs, lr, model, train_loader, test_loader, opt_func)

# torch.save(model.state_dict(), model_weights_path)

# model.load_state_dict(torch.load(model_weights_path))
# model.eval()


# t = 0

# for i in range(test_size):
#     fpath = test_paths[i]
#     image = load_image(fpath)
#     image = image.transpose((2, 0, 1))
#     image = torch.tensor(image, dtype=torch.float32).to(device)
#     attrs = test_genders[i]
#     attrs = np.array([1, 0]) if attrs == 0 else np.array([0, 1])
#     attrs = torch.tensor(attrs, dtype=torch.float32).to(device)
    
    
    
#     image = image.unsqueeze(0)
#     attrs = attrs.unsqueeze(0)
    
#     # print(image.shape)
#     # print(attrs.shape)
    
#     out_data = model(image)
#     out_data = out_data.tolist()[0]
#     max = np.argmax(out_data)
#     out_data[max] = 1
#     out_data[out_data != 1] = 0
    
#     if attrs.tolist()[0] == out_data:
#         t += 1
    
    


# print(t/test_size)

# for i, sample in enumerate(train_dataset):
#     print(i, sample[0].shape, sample[1].shape)
#     if i == 10:
#         break



## -----------------------------------------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------------------------------------
def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step

model = models.resnet18(pretrained=True)
for params in model.parameters():
  params.requires_grad_ = False

nr_filters = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(nr_filters, 2), nn.Softmax())

model = model.to(device)

# loss_fn = nn.modules.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.fc.parameters()) 

# train_step = make_train_step(model, optimizer, loss_fn)



# losses = []
# val_losses = []

# epoch_train_losses = []
# epoch_test_losses = []

# n_epochs = 10
# early_stopping_tolerance = 3
# early_stopping_threshold = 0.03

# for epoch in range(n_epochs):
#   epoch_loss = 0
#   for i ,data in tqdm(enumerate(train_loader), total = len(train_loader)): #iterate ove batches
#     x_batch , y_batch = data
#     x_batch = x_batch.to(device) #move to gpu
#     # y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
#     y_batch = y_batch.to(device) #move to gpu


#     loss = train_step(x_batch, y_batch)
#     epoch_loss += loss/len(train_loader)
#     losses.append(loss)
    
#   epoch_train_losses.append(epoch_loss)
#   print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

#   #validation doesnt requires gradient
#   with torch.no_grad():
#     cum_loss = 0
#     for x_batch, y_batch in train_loader:
#       x_batch = x_batch.to(device)
#     #   y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
#       y_batch = y_batch.to(device)

#       #model to eval mode
#       model.eval()

#       yhat = model(x_batch)
#       val_loss = loss_fn(yhat,y_batch)
#       cum_loss += loss/len(train_loader)
#       val_losses.append(val_loss.item())


#     epoch_test_losses.append(cum_loss)
#     print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  
    
#     best_loss = min(epoch_test_losses)
    
#     #save best model
#     if cum_loss <= best_loss:
#       best_model_wts = model.state_dict()
    
#     #early stopping
#     early_stopping_counter = 0
#     if cum_loss > best_loss:
#       early_stopping_counter +=1

#     if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
#       print("/nTerminating: early stopping")
#       break #terminate training
    
# #load best model
# model.load_state_dict(best_model_wts)
# torch.save(model.state_dict(), model_weights_path)



# model.load_state_dict(torch.load(model_weights_path))
# model.eval()

# t = 0

# for i in range(test_size):
#     fpath = test_paths[i]
#     image = load_image(fpath)
#     image = image.transpose((2, 0, 1))
#     image = torch.tensor(image, dtype=torch.float32).to(device)
#     attrs = test_genders[i]
#     attrs = np.array([1, 0]) if attrs == 0 else np.array([0, 1])
#     attrs = torch.tensor(attrs, dtype=torch.float32).to(device)
    
    
    
#     image = image.unsqueeze(0)
#     attrs = attrs.unsqueeze(0)
    
#     # print(image.shape)
#     # print(attrs.shape)
    
#     out_data = model(image)
#     out_data = out_data.tolist()[0]
#     max = np.argmax(out_data)
#     out_data[max] = 1
#     out_data[out_data != 1] = 0
    
#     if attrs.tolist()[0] == out_data:
#         t += 1
    
    


# print(t/test_size)
