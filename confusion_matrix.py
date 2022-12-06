# confusion matrices

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
import matplotlib.pyplot as plt

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

    num_train = len(open("train_labels.csv", 'r').readlines())
    num_val = len(open("val_labels.csv", 'r').readlines())
    num_test = len(open("test_labels.csv", 'r').readlines())

    print("num train ", num_train, " num val ", num_val, " num test ", num_test)

    dataset_sizes = {'train' : 0, 'val' : 0, 'test' : num_test}

    # train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    # val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = num_test, shuffle = False)
    dataloaders = {'train' : 0, 'val' : 0, 'test' : test_dataloader}

    return dataloaders, dataset_sizes

def get_preds_and_labels(model, dataloaders):
    phase = 'test'
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(preds)
        print(preds.shape)

        print(labels, labels.shape)

    return preds, labels

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def confusion_matrix_func(model_name, preds, true, classes):
    mat = confusion_matrix(true, preds)

    # plt.clf()
    # plt.matshow(mat)
    # plt.xticks(range(len(classes)), classes)
    # plt.yticks(range(len(classes)), classes)
    # # plt.colorbar()
    # plt.title(f"Confusion Matrix for {model_name}")
    # plt.show()

    f = open("confusion_matrix/precision_recall.txt", "a")
    
    print("Model Name: ", model_name)

    precision = precision_score(true, preds, average = "weighted")
    recall = recall_score(true, preds, average = "weighted")
    print("Weighted Precision: ", precision)
    print("Weighted Recall: ", recall)

    f.write(f"\n\nModel Name: {model_name}")
    f.write(f"\nWeighted Precision: {precision}")
    f.write(f"\nWeighted Recall: {recall}")

    precision = precision_score(true, preds, average = "micro")
    recall = recall_score(true, preds, average = "micro")
    print("Micro Precision: ", precision)
    print("Micro Recall: ", recall)
    f.write(f"\nMicro Precision: {precision}")
    f.write(f"\nMicro Recall: {recall}")

    precision = precision_score(true, preds, average = "macro")
    recall = recall_score(true, preds, average = "macro")
    print("Macro Precision: ", precision)
    print("Macro Recall: ", recall)
    f.write(f"\nMacro Precision: {precision}")
    f.write(f"\nMacro Recall: {recall}")

    f.close()


    # plt.savefig(f"confusion_matrix/{model_name}_small.png")

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
    # print(model)

    # remove num classes layer
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

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
    # print(model)
    # remove num classes layer
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

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
    # print(model)
    # remove num classes layer
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

# EfficientNet_b4
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

    best_model_wts_pth = model_name + ".pth"

    model.load_state_dict(torch.load(best_model_wts_pth))
    # print(model)
    # remove num classes layer
    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

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

    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

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

    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders

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

    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders

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

    # model.fc = nn.Sequential(*list(model.fc.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders

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

    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders

metadata_path = "data/archive(2)/HAM10000_metadata.csv"
df = pd.read_csv(metadata_path, sep = ",")
print(df)
classes = sorted(df.dx.unique())
print(classes)

num_epochs = 100
num_classes = 7

# AlexNet
model, dataloaders = AlexNet(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("AlexNet", preds, labels, classes)

model, dataloaders = EfficientNet_b3(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("EfficientNetB3", preds, labels, classes)

model, dataloaders = EfficientNet_b4(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("EfficientNetB4", preds, labels, classes)

model, dataloaders = EfficientNet_b5(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("EfficientNetB5", preds, labels, classes)

model, dataloaders = MobileNetV2(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("MobileNetV2", preds, labels, classes)

model, dataloaders = VGG16(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("VGG16", preds, labels, classes)

model, dataloaders = VGG19(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("VGG19", preds, labels, classes)

model, dataloaders = InceptionV3(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("InceptionV3", preds, labels, classes)

model, dataloaders = DenseNet121(num_classes, num_epochs)
preds, labels = get_preds_and_labels(model, dataloaders)
confusion_matrix_func("DenseNet121", preds, labels, classes)