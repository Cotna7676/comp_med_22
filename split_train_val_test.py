# split labels.csv into train_labels.csv, val_labels.csv, test_labels.csv

import pandas as pd
import numpy as np

df = pd.read_csv("labels.csv", sep = ",", header = None)
# df = np.loadtxt("labels.csv", delimiterstr=",", header = None)
print(df)

num_classes = 7

train_frac, val_frac, test_frac = (0.6, 0.2, 0.2)

train_df = None
val_df = None
test_df = None

# for each class, get 60, 20, 20 sampling for train, val, test
for i in range(num_classes):
    df_by_class = df.loc[df[1] == i]
    print(df_by_class)
    print(len(df_by_class))

    indices = np.arange(len(df_by_class))
    np.random.shuffle(indices)
    # print(indices)

    train_amt = int(len(indices) * train_frac)
    val_amt = int(len(indices) * val_frac)
    test_amt = len(indices) - train_amt - val_amt

    # print(train_amt, val_amt, test_amt, train_amt + val_amt + test_amt)

    train_part = indices[:train_amt]
    val_part = indices[train_amt:train_amt+val_amt]
    test_part = indices[train_amt+val_amt:]
    print(len(train_part), len(val_part), len(test_part))
    # print(train_part)

    train_part = df_by_class.iloc[train_part]
    val_part = df_by_class.iloc[val_part]
    test_part = df_by_class.iloc[test_part]

    if i == 0:
        train_df = train_part
        val_df = val_part
        test_df = test_part
    else:
        train_df = pd.concat([train_df, train_part], axis = 0)
        val_df = pd.concat([val_df, val_part], axis = 0)
        test_df = pd.concat([test_df, test_part], axis = 0)

print(len(train_df), len(val_df), len(test_df), len(train_df) +len(val_df) +len(test_df))
print(len(df))

train_df.to_csv("train_labels.csv", sep = ",", index = False, header = False)
val_df.to_csv("val_labels.csv", sep = ",", index = False, header = False)
test_df.to_csv("test_labels.csv", sep = ",", index = False, header = False)