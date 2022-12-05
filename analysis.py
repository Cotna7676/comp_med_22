# analysis

import numpy as np
import matplotlib.pyplot as plt

# code used from 10417
def plot_curves(question_part, train_loss, train_acc, test_loss, test_acc, epoch_num):
    epochs = list(np.arange(epoch_num))

    plt.clf()
    plt.plot(epochs, train_loss, label = "Train Loss")
    plt.plot(epochs, test_loss, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{question_part} Loss over Epochs")
    plt.savefig(f"training_plots/{question_part}_loss.png")
    #plt.show()

    plt.clf()
    plt.plot(epochs, train_acc, label = "Train Accuracy")
    plt.plot(epochs, test_acc, label = "Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{question_part} Accuracy over Epochs")
    plt.savefig(f"training_plots/{question_part}_acc.png")
    #plt.show()

def parse_log_txt(path):
    file = open(path, "r")
    lines = file.readlines()
    file.close()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for i in range(100):
        line = lines[i]
        split = line.split(",")
        epoch_num, train_loss, train_acc, val_loss, val_acc = split

        train_loss_list.append(float(train_loss))
        train_acc_list.append(float(train_acc))
        val_loss_list.append(float(val_loss))
        val_acc_list.append(float(val_acc))
    
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

def individual_plots(list_of_models):
    for model_tuple in list_of_models:
        name, model_path = model_tuple
        print(name)
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = parse_log_txt(model_path)

        plot_curves(name, train_loss_list, train_acc_list, val_loss_list, val_acc_list, 100)

def combo_plots(list_of_models):
    epochs = list(np.arange(100))

    plt.clf()
    # Plot Loss
    for model_tuple in list_of_models:
        name, model_path = model_tuple
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = parse_log_txt(model_path)

        plt.plot(epochs, train_loss_list, label = f"{name} Train Loss")
        plt.plot(epochs, val_loss_list, label = f"{name} Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss over Epochs")
    plt.savefig(f"training_plots/combo_models_loss.png")
    plt.show()

    plt.clf()
    # Plot Acc
    for model_tuple in list_of_models:
        name, model_path = model_tuple
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = parse_log_txt(model_path)

        plt.plot(epochs, train_acc_list, label = f"{name} Train Acc")
        plt.plot(epochs, val_acc_list, label = f"{name} Validation Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy over Epochs")
    plt.savefig(f"training_plots/combo_models_acc.png")
    plt.show()

def combo_plots_4_plots(list_of_models):
    epochs = list(np.arange(100))

    # plt.clf()
    # # Plot Loss
    # for model_tuple in list_of_models:
    #     name, model_path = model_tuple
    #     train_loss_list, train_acc_list, val_loss_list, val_acc_list = parse_log_txt(model_path)

    #     plt.plot(epochs, train_loss_list, label = f"{name} Train Loss")
    #     plt.plot(epochs, val_loss_list, label = f"{name} Validation Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title(f"Loss over Epochs")
    # plt.savefig(f"training_plots/combo_models_loss.png")
    # plt.show()

    plt.clf()
    # Plot Train
    for model_tuple in list_of_models:
        name, model_path = model_tuple
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = parse_log_txt(model_path)

        plt.plot(epochs, train_acc_list, label = f"{name} Train Acc")
        # plt.plot(epochs, val_acc_list, label = f"{name} Validation Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Train Accuracy over Epochs")
    plt.savefig(f"training_plots/combo_models_train_acc.png")
    # plt.show()

    plt.clf()
    # Plot Val Acc
    for model_tuple in list_of_models:
        name, model_path = model_tuple
        train_loss_list, train_acc_list, val_loss_list, val_acc_list = parse_log_txt(model_path)

        # plt.plot(epochs, train_acc_list, label = f"{name} Train Acc")
        plt.plot(epochs, val_acc_list, label = f"{name} Validation Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Validation Accuracy over Epochs")
    plt.savefig(f"training_plots/combo_models_val_acc.png")
    # plt.show()

def main():
    artifacts_path = "artifacts/"
    AlexNet = ("AlexNet", artifacts_path + "AlexNet__epochs_100.log.txt")
    EfficientNetB3 = ("EfficientNetB3", artifacts_path + "EfficientNetB3__epochs_100.log.txt")
    EfficientNetB4 = ("EfficientNetB4", artifacts_path + "EfficientNetB4__epochs_100.log.txt")
    EfficientNetB5 = ("EfficientNetB5", artifacts_path + "EfficientNetB5__epochs_100.log.txt")
    MobileNetV2 = ("MobileNetV2", artifacts_path + "MobileNetV2__epochs_100.log.txt")
    VGG16 = ("VGG16", artifacts_path + "VGG16__epochs_100.log.txt")
    VGG19 = ("VGG19", artifacts_path + "VGG19__epochs_100.log.txt")

    model_list = [AlexNet, EfficientNetB3, EfficientNetB4, EfficientNetB5, MobileNetV2, VGG16, VGG19]

    # individual_plots(model_list)
    # combo_plots(model_list)
    combo_plots_4_plots(model_list)

main()

