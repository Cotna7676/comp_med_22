## Comp Medicine Project GitHub

1. Download dataset from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download
2. ```python3 parse_metadata.py``` # parse metadata and create labels.csv
3. ```python3 split_train_val_test.py``` # split labels.csv to train, val, test subsets
    The split is 60-20-20 for each of the 7 classes.