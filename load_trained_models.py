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

def save_test_loss_acc(model_name, test_loss, test_acc):
    f = open(f"{model_name}.log.txt", "a")
    f.write(f"Best Model Test Loss/Acc:{test_loss},{test_acc}")
    f.close()

# train model
def run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes):
    model.eval()
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

# initialize datasets and dataloaders
def initialize_dataloaders(img_size, crop_size):
    # https://pytorch.org/hub/pytorch_vision_alexnet/
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
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
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

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

    best_model_wts_pth = model_name + ".pth"

    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# EfficientNet_b3
def EfficientNet_b3(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

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

    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# EfficientNet_b4
def EfficientNet_b4(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

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

    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# EfficientNet_b5
def EfficientNet_b5(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

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
    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# MobileNetV2
def MobileNetV2(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

    model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier = nn.Linear(1280,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "MobileNetV2"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    print(model_desc)
    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# VGG16
def VGG16(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

    model = torchvision.models.vgg16(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier[-1] = nn.Linear(4096,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "VGG16"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    print(model_desc)
    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# VGG19
def VGG19(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

    model = torchvision.models.vgg19(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier[-1] = nn.Linear(4096,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "VGG19"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    print(model_desc)
    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# InceptionV3
def InceptionV3(num_classes, num_epochs):
    img_size = 299
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 299)

    model = torchvision.models.inception_v3(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.aux_logits=False # https://stackoverflow.com/questions/51045839/pytorch-inceptionv3-transfer-learning-gives-error-max-received-an-invalid-co
    model.fc = nn.Linear(2048,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)
    print(model.parameters)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "InceptionV3"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    print(model_desc)
    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

# DenseNet121
def DenseNet121(num_classes, num_epochs):
    img_size = 256
    dataloaders, dataset_sizes = initialize_dataloaders(img_size, 224)

    model = torchvision.models.densenet121(weights='IMAGENET1K_V1')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    print(model.parameters)
    # have output be number of classes
    model.classifier = nn.Linear(1024,num_classes)
    # print(model)
    # print(model.parameters)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_desc = "DenseNet121"
    model_name = "artifacts/" + model_desc + "__epochs_" + str(num_epochs)

    print(model_desc)
    best_model_wts_pth = model_name + ".pth"
    model.load_state_dict(torch.load(best_model_wts_pth))

    run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

num_epochs = 100
num_classes = 7
# AlexNet(num_classes, num_epochs)
# EfficientNet_b3(num_classes, num_epochs)
# EfficientNet_b4(num_classes, num_epochs)
# EfficientNet_b5(num_classes, num_epochs)
# MobileNetV2(num_classes, num_epochs)
# VGG16(num_classes, num_epochs)
# VGG19(num_classes, num_epochs)
# InceptionV3(num_classes, num_epochs)
DenseNet121(num_classes, num_epochs)
