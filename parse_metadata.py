# parse metadata and create labels.csv

import pandas as pd
import csv
import os

metadata_path = "data/archive(2)/HAM10000_metadata.csv"

df = pd.read_csv(metadata_path, sep = ",")
print(df)

classes = sorted(df.dx.unique())
# print(classes)

class_label_list = [i for i in range(len(classes))]
# print(class_label_list)

mapping = dict(zip(classes, class_label_list))
print(mapping)

# create annotations file
# data
# |- archive(2)
#    |- ...part_1
#    |- ...part_2

img_type = ".jpg"

with open('labels.csv', 'w') as f:
    writer = csv.writer(f)

    for index, row in df.iterrows():
        # print(row.image_id, mapping[row.dx])

        part1_path = "data/archive(2)/HAM10000_images_part_1/"
        part2_path = "data/archive(2)/HAM10000_images_part_2/"

        part1_img_path = part1_path + row.image_id + img_type
        part2_img_path = part2_path + row.image_id + img_type

        if os.path.exists(part1_img_path):
            data = [part1_img_path, mapping[row.dx]]
        elif os.path.exists(part2_img_path):
            data = [part2_img_path, mapping[row.dx]]
        else:
            raise Exception

        writer.writerow(data)
