#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_splitter.py
# @Author: Yin-tao Xu
# @Date  : 18-9-8
# @Desc  : This scipt aims at splitting the data into test set and trainning set by
#  the ratio of 2: 8. How to split is not the core topic of this project, therefore,
#  it is high time that we could rely on scikit-learn.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from utils import save
import sys

def split_data():
    dataset_path = os.path.join('.', 'data', 'insurance.csv')
    if not os.path.exists(dataset_path):
        print("Please download dataset from: ")
        print(r"https://www.kaggle.com/mirichoi0218/insurance")
        print("After that, put insurance.csv under: ")
        print(os.path.abspath(dataset_path))
        sys.exit(0)
    df = pd.read_csv(dataset_path)

    y = df.iloc[:, 4].values
    x = np.concatenate((df.iloc[:, 0: 4].values, df.iloc[:, 5:].values), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    p_train_x = os.path.join(".", "data", "x_train.pkl")
    p_train_y = os.path.join(".", "data", "y_train.pkl")
    p_test_x = os.path.join(".", "data", "x_test.pkl")
    p_test_y = os.path.join(".", "data", "y_test.pkl")

    save(x_train, p_train_x)
    save(y_train, p_train_y)
    save(x_test, p_test_x)
    save(y_test, p_test_y)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    split_data()
