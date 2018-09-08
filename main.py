#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: Yin-tao Xu
# @Date  : 18-9-8
# @Desc  : Show the main process of Bayes estimation

from data_spliter import split_data
from model import Naive_bayes
import numpy as np
from visualize import plot_computation_process, output_basic_info

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = split_data()

    model = Naive_bayes()
    # we are going to split dimension of age, BMI and charges
    model.fit(X_train, y_train, discretize_dim=(0, 2, 5), split_num=(5, 5, 5))

    pred, _, _ = model.predict(X_train)
    train_acc = np.sum((pred == y_train)) / y_train.shape[0]

    pred, _, _ = model.predict(X_test)
    test_acc = np.sum((pred == y_test)) / y_test.shape[0]
    print("train accuracy: {}, test accuracy: {}".format(train_acc, test_acc))

    # chekc for the sample
    sample_index = 8
    output_basic_info(X_test[sample_index])
    plot_computation_process(model, X_test[sample_index])
