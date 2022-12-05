
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

def get_features(model, dataloaders):
    phase = 'test'
    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        print(outputs)
        print(outputs.shape)

        print(labels, labels.shape)

    return outputs, labels

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_tsne(model_name, outputs, labels, classes):
    n_components = 2
    tsne = TSNE(n_components, learning_rate = 'auto')
    tsne_result = tsne.fit_transform(outputs)
    # pca = PCA(n_components=2)
    # tsne_result = pca.fit_transform(outputs)
    # print(tsne_result.shape)

    # print(labels)
    plt.clf()
    for i in range(len(classes)):
        class_labels = torch.nonzero(torch.where(labels == i, 1, 0))
        # print(class_labels.shape)
        # print(labels[class_labels].shape)
        # print(i)
        # print(tsne_result[class_labels].shape)
        squeezed = np.squeeze(tsne_result[class_labels])
        # print(squeezed.shape)
        # print(tsne_result[class_labels])
        # print(squeezed[:, 0], squeezed[:, 1])
        plt.scatter(squeezed[:, 0], squeezed[:, 1], label = str(classes[i]))
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.title(f"TSNE Plot for {model_name} Features")
    plt.legend()
    plt.show()

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
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

    # run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

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
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders
    # run_train_model(model, model_name, criterion, optimizer, num_epochs, dataloaders, dataset_sizes)

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
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
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
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
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

    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    print(model)

    model.eval()

    return model, dataloaders

    # run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

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

    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders

    # run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

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

    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders
    # run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

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

    model.fc = nn.Sequential(*list(model.fc.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders
    # run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

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

    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
    print(model)

    model.eval()

    return model, dataloaders

    # run_train_model(model, model_name, criterion, best_model_wts_pth, dataloaders, dataset_sizes)

metadata_path = "data/archive(2)/HAM10000_metadata.csv"
df = pd.read_csv(metadata_path, sep = ",")
print(df)
classes = sorted(df.dx.unique())
print(classes)

num_epochs = 100
num_classes = 7

# AlexNet
# model, dataloaders = AlexNet(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("AlexNet", outputs, labels, classes)

# model, dataloaders = EfficientNet_b3(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("EfficientNet_b3", outputs, labels, classes)

# model, dataloaders = EfficientNet_b4(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("EfficientNet_b4", outputs, labels, classes)

# model, dataloaders = EfficientNet_b5(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("EfficientNet_b5", outputs, labels, classes)

# model, dataloaders = MobileNetV2(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("MobileNetV2", outputs, labels, classes)

# model, dataloaders = VGG16(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("VGG16", outputs, labels, classes)

# model, dataloaders = VGG19(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("VGG19", outputs, labels, classes)

# model, dataloaders = InceptionV3(num_classes, num_epochs)
# outputs, labels = get_features(model, dataloaders)
# get_tsne("InceptionV3", outputs, labels, classes)

model, dataloaders = DenseNet121(num_classes, num_epochs)
outputs, labels = get_features(model, dataloaders)
get_tsne("DenseNet121", outputs, labels, classes)