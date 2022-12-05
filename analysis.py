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
    plt.savefig(f"{question_part}_loss.png")
    #plt.show()

    plt.clf()
    plt.plot(epochs, train_acc, label = "Train Accuracy")
    plt.plot(epochs, test_acc, label = "Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{question_part} Accuracy over Epochs")
    plt.savefig(f"{question_part}_acc.png")
    #plt.show()


def main():
    artifacts_path = "artifacts/"
    AlexNet = ("AlexNet", artifacts_path + "AlexNet__epochs_100.log.txt")
    EfficientNetB3 = ("EfficientNetB3", artifacts_path + "EfficientNetB3__epochs_100.log.txt")
    EfficientNetB4 = ("EfficientNetB4", artifacts_path + "EfficientNetB4__epochs_100.log.txt")
    EfficientNetB5 = ("EfficientNetB5", artifacts_path + "EfficientNetB5__epochs_100.log.txt")
    MobileNetV2 = ("MobileNetV2", artifacts_path + "MobileNetV2__epochs_100.log.txt")

    