import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
from torchmetrics import F1Score

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_labels.iloc[idx, 0])
        img_path = self.img_labels.iloc[idx, 0]
        #image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

from torchvision import transforms, utils
import copy, time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0
            f_score_corrects = 0

            num_batches = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                num_batches += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                f1 = F1Score(num_classes = 7)
                f_score_corrects += f1(preds, labels.data)
                
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # f score
            epoch_acc = f_score_corrects.double() / num_batches

            if phase == 'train':
                train_loss[epoch] = epoch_loss
                train_acc[epoch] = epoch_acc
            else:
                val_loss[epoch] = epoch_loss
                val_acc[epoch] = epoch_acc

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_model_wts, train_loss, train_acc, val_loss, val_acc

# save loss accuracy in txt files
def save_loss_acc(model_name, num_epochs, train_loss, train_acc, val_loss, val_acc):
    f = open(f"{model_name}.log.txt", "w")
    for epoch in range(num_epochs):
        train_loss_val = train_loss[epoch]
        train_acc_val = train_acc[epoch]
        val_loss_val = val_loss[epoch]
        val_acc_val = val_acc[epoch]
        f.write(f"{epoch},{train_loss_val},{train_acc_val},{val_loss_val},{val_acc_val}\n")
    f.close()

def save_test_loss_acc(model_name, test_loss, test_acc):
    f = open(f"{model_name}.log.txt", "a")
    f.write(f"Best Model Test Loss/Acc:{test_loss},{test_acc}")
    f.close()

# train model
def run_train_model(model, model_name, criterion, optimizer, num_epochs, dataloaders, dataset_sizes):
    # train model, get train/val acc/loss
    model, best_model_wts, train_loss, train_acc, val_loss, val_acc =\
                 train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes)

    save_loss_acc(model_name, num_epochs, train_loss, train_acc, val_loss, val_acc)

    best_model = model.load_state_dict(best_model_wts)
    phase = 'test'
    running_loss = 0
    running_corrects = 0
    f_score_corrects = 0

    num_batches = 0
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        num_batches += 1

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        f1 = F1Score(num_classes = 7)
        f_score_corrects += f1(preds, labels.data)

    test_loss = running_loss / dataset_sizes[phase]
    # test_acc = running_corrects.double() / dataset_sizes[phase]

    # f score
    test_acc = f_score_corrects.double() / num_batches

    print(test_loss, test_acc)
    save_test_loss_acc(model_name, test_loss, test_acc)

    # save model
    torch.save(model.state_dict(), model_name + ".pth")

# initialize datasets and dataloaders
def initialize_dataloaders(img_size):
    # https://pytorch.org/hub/pytorch_vision_alexnet/
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(224),
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomImageDataset("train_labels.csv", preprocess)
    val_dataset = CustomImageDataset("val_labels.csv", preprocess)
    test_dataset = CustomImageDataset("test_labels.csv", preprocess)

    batch_size = 64
    shuffle = True

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    dataloaders = {'train' : train_dataloader, 'val' : val_dataloader, 'test' : test_dataloader}
    num_train = len(open("train_labels.csv", 'r').readlines())
    num_val = len(open("val_labels.csv", 'r').readlines())
    num_test = len(open("test_labels.csv", 'r').readlines())

    print("num train ", num_train, " num val ", num_val)

    dataset_sizes = {'train' : num_train, 'val' : num_val, 'test' : num_test}

    return dataloaders, dataset_sizes


# AlexNet -- not modified
def AlexNet(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size)

    model = torchvision.models.alexnet(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier[6] = nn.Linear(4096,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "AlexNet"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    run_train_model(model, model_name, criterion, optimizer, num_epochs, dataloaders, dataset_sizes)

# EfficientNet_b3
def EfficientNet_b3(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size)

    model = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier[-1] = nn.Linear(1536,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "EfficientNetB3"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    run_train_model(model, model_name, criterion, optimizer, num_epochs, dataloaders, dataset_sizes)

# EfficientNet_b4
def EfficientNet_b4(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size)

    model = torchvision.models.efficientnet_b4(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier[-1] = nn.Linear(1792,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "EfficientNetB4"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    run_train_model(model, model_name, criterion, optimizer, num_epochs, dataloaders, dataset_sizes)

# EfficientNet_b5
def EfficientNet_b5(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size)

    model = torchvision.models.efficientnet_b5(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier[-1] = nn.Linear(2048,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "EfficientNetB5"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    print(model_desc)

    run_train_model(model, model_name, criterion, optimizer, num_epochs, dataloaders, dataset_sizes)

num_epochs = 100
num_classes = 7
# AlexNet(num_classes, num_epochs)
# EfficientNet_b3(num_classes, num_epochs)
# EfficientNet_b4(num_classes, num_epochs)
EfficientNet_b5(num_classes, num_epochs)