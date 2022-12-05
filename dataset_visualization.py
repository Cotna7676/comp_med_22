# get plot of number of images in classes

import pandas as pd
import csv
import os
import matplotlib.pyplot as plt

metadata_path = "data/archive(2)/HAM10000_metadata.csv"

df = pd.read_csv(metadata_path, sep = ",")
print(df)

classes = sorted(df.dx.unique())
print(classes)

class_label_list = [i for i in range(len(classes))]
# print(class_label_list)

mapping = dict(zip(classes, class_label_list))
print(mapping)

def get_bar_plt_num_imgs_per_class():
    num_imgs = [0 for i in range(len(classes))]

    for index, row in df.iterrows():
        # print(row.dx, mapping[row.dx])

        val = mapping[row.dx]

        num_imgs[val] += 1

    print(num_imgs, sum(num_imgs))

    plt.bar(classes, num_imgs)
    plt.xlabel("Skin Lesion Cancer Classes")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Skin Lesion Cancer Class in MNIST:HAM10000")
    plt.show()

def get_imgs_per_class():

    num_imgs = [0 for i in range(len(classes))]

    part1_path = "data/archive(2)/HAM10000_images_part_1/"
    part2_path = "data/archive(2)/HAM10000_images_part_2/"
    img_type = ".jpg"

    for index, row in df.iterrows():
        part1_img_path = part1_path + row.image_id + img_type
        part2_img_path = part2_path + row.image_id + img_type

        if os.path.exists(part1_img_path):
            data = [part1_img_path, mapping[row.dx]]
        elif os.path.exists(part2_img_path):
            data = [part2_img_path, mapping[row.dx]]
        else:
            raise Exception

        val = data[1]
        if num_imgs[val] == 0:
            num_imgs[val] = data[0]

    from PIL import Image
    import cv2

    fig = plt.figure(figsize=(10, 7))
    rows = 2
    cols = 4

    #subplot(r,c) provide the no. of rows and columns
    print(num_imgs)
    for i in range(len(num_imgs)):
        img_path = num_imgs[i]

        # im = Image.open(img_path)
        # im.save(f"dataset_visualizations/{i}.jpg")
        # im.show()

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(classes[i])

    plt.tight_layout()
    plt.show()


test_path = "test_labels.csv"

file = open(test_path)
lines = file.readlines()
file.close()

num_imgs = [0 for i in range(len(classes))]

for line in lines:
    split = int(line.split(",")[1])
    num_imgs[split] += 1

plt.bar(classes, num_imgs)
plt.xlabel("Skin Lesion Cancer Classes")
plt.ylabel("Number of Images")
plt.title("Number of Images per Skin Lesion Cancer Class in Test Set")
plt.show()