#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: Yin-tao Xu
# @Date  : 18-9-8
# @Desc  : This module aims at providing interface to do the naive bayesian estimation.
# Sketch of interfaces in this file:
# 'Naive_bayes' is a class to do the naive bayes estimation on this
# specified smoking predicted question. Two functions are of significance:
#            1.fit(X, y)
#                to fit the data
#            2.pred, prob = predict(X)
#                to predict, output the posterior probability for positive result
#                and final estimation result
# apply_raw2index: depend on Naive_bayes.feat2index_dict_x which is generated during
#           tranning process , we can transcribe the raw feature into integers as indexes.
#           I make it as interface since it is one of the necessities to do the
#           visualization (we need to access inner variables rather than its output).

import os
import numpy as np
import copy
import math
from utils import load

__all__ = ["Naive_bayes", "apply_raw2index"]

class Discretizer():
    """
    This class is responsible for discretizing the continious input random varible,
    since our bayesian estimation bases on the discreted cases. The basic method
    is to equally split the range of the target input dimension.
    """
    def fit(self, X, todo_index, split_num):
        """
        Fit the input data(X) by specified parameters provided by todo_index and
         split num
        :param X: numpy array, a design matrix, with shape (num_input x dimension_feature)
        :param todo_index: a list, of which each element represents the dimension you want to
        discretize
        :param split_num: a list, one element in todo_index corresponds to one element here,
        which represents the quantities of intervals you are willing to split
        :return: None
        """

        feat_mins = []
        intervals = []

        for k, i in enumerate(todo_index):
            num = split_num[k]
            feat_min = np.min(X[:, i])
            feat_mins.append(feat_min)
            feat_max = np.max(X[:, i])
            interval = (feat_max - feat_min) / num
            intervals.append(interval)

        # todi_index is exactly the same as explained in function description
        self.todo_index = todo_index
        # split_num is exactly the same as explained in function description
        self.split_num = split_num
        # feat_min is a list, of which each element is
        # the minimum value in each target dimension
        self.feat_mins = feat_mins
        # intervals is a list, of which each element is
        # the interval = (feat_max - feat_min) / num
        self.intervals = intervals

    def apply(self, X):
        """
        apply the discretization operation on the input designed matrix
        (directly modify the input matrix)
        :param X: numpy array, a design matrix, with shape (num_input x dimension_feature)
        :return: None
        """
        for k, i in enumerate(self.todo_index):
            num = self.split_num[k]
            X[:, i] -= self.feat_mins[k]
            X[:, i] = X[:, i] / self.intervals[k]
            X[:, i] = np.floor(np.array(X[:, i], dtype=np.float)).astype(np.int)
            X[:, i] = np.minimum(np.maximum(X[:, i], 0), num - 1)

def feat2index(rawx_column):
    """
    Map each value occured in the column into an integer
    :param rawx_column:  numpy array, a column of
    design matrix, with shape (num_input x 1)
    :return: a dictionary, which maps each value occuered in the column into an
    integer as the index. (e.g: {'male': 0, 'female': 1})
    """
    i = 0
    ret_dict = {}
    for e in rawx_column:
        try:
            ret_dict[e]
        except KeyError:
            ret_dict[e] = i
            i += 1
    return ret_dict

def raw2index(raw_matrix, discretizer=None):
    """
    Map each value occured in the columns of designed matrix
     into an integer. If there is a discretor, transcribe the columns
     discretized by it into indexes directly base on minimum and intervals(
     provided by discretor)
    :param raw_matrix: numpy array, a design matrix, with shape (num_input x dimension_feature)
    :param discretizer: a Discretizer defined above
    :return: a list of dictionaries, which maps each value occuered in the object dimension
     into an integer as the index. Specially, those dimensions are requied to be discretizered
     need to be preprocessed by discretizer. In fact, they have been transformed into indexes.
     Due to consistency of each dimension, we retranscribe it basing on indexes provided by
     discretor.
     (example for a dictionary: {'male': 0, 'female': 1})
     (example for an output: [dict1, dict2, dict3, ..., dictn] with each element
     is a dictionary in the form of above)
    """

    feat_dicts = [] # a placeholder for dictionaries generated as follows
    for i in range(raw_matrix.shape[1]):
        # if user inputs a discretizer, than treat discretized feature specially
        if discretizer:
            if i in discretizer.todo_index:
                index = discretizer.todo_index.index(i)
                # Note: why i do this because it is worth noting that some indexes
                # will not occur in trainning data since value at some intervals may
                #  be missing if we treat it same like other dimension. When the test
                #  data happens to be these intervals, bugs will occur.
                # One advisable way is to generate it directly from discretizer
                now_dict = feat2index([_ for _ in range(discretizer.split_num[index])])
            else:
                now_dict = feat2index(raw_matrix[:, i])
        else:
            now_dict = feat2index(raw_matrix[:, i])
        feat_dicts.append(now_dict)
    return feat_dicts

def apply_raw2index(raw_matrix, feat2index_dict_x):
    """
    Apply the transformation of a raw_matrix into a matrix, of which each element
    is the index of all possible values generated by raw2index function.
    :param raw_matrix: numpy array, a design matrix, with shape (num_input x dimension_feature)
    :param feat2index_dict_x: exactly the output of function 'raw2index'
    see the introduction of its output for help
    :return: None(do the modification of the input matrix directly)
    """
    for i in range(raw_matrix.shape[1]):
        for j in range(raw_matrix.shape[0]):
            raw_matrix[j, i] = feat2index_dict_x[i][raw_matrix[j, i]]


def inverst_dict(d):
    """
    Inverse a dictionary. (The key of new dict. is the val. of the original one. Similar
    to the value of new.dict)
    :param d: a python dictionary
    :return: a inversed python dictionary
    """
    return dict(zip(d.values(), d.keys()))

def apply_index2raw(y, feat2index_dict_y):
    """
    The reverse process of apply_raw2index, which is realized due to we should recover
    the y's value from indexes to original input value. (e.g: 0 -> 'yes', 1 -> 'no' for
    this specified smoking prediction problem).
    Note: we do not directly modify y here(it seems to be nonsense, but it is pratical).
    :param y: a 1-d numpy array, of which each element is the output index value
    :param feat2index_dict_y: a dictionary generated by feat2index
    :return: a recoverd 1-d numpy array from indexes of value into values
    """
    inv_dict = inverst_dict(feat2index_dict_y)
    ret = []
    for i in y:
        ret.append(inv_dict[i])
    return ret

class Naive_bayes():
    """
    The core section do the Bayes estimation rather than doing feature engineering
    trivals like functions above.
    """
    def fit(self, X, y, discretize_dim=(), split_num=()):
        """
        Train the model with input X, y. And do the discretization of the dimension
        specified in discretize_dim by splitting the range into equal intervals. The
        quantities of interval is specified in split num.
        :param X: numpy array, a design matrix, with shape (num_input x dimension_feature)
        :param y: 1-d numpy array, (num_input x 1)
        :param discretize_dim: a list of dimensions indexes awaiting to be discretized
        :param split_num: a list, one element in discretize dim
        corresponds to one element here, which represents the quantities of intervals
         you are willing to split.
        :return: None
        """
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)

        # discretize age, BMI and charges with equal interval at
        # this specified task
        assert len(discretize_dim) == len(split_num), "assure each discretization" \
                                                      " dimension has specified num of split"

        # train the discretizer and transform the input continious dimension into
        # discrete indexes
        d = Discretizer()
        d.fit(X, discretize_dim, split_num)
        d.apply(X)

        # turn all possible values in each dimension to its corresponding indexes
        # generated by raw2index function
        feat2index_dict_x = raw2index(X, d)
        apply_raw2index(X, feat2index_dict_x)

        # similarity to y, we reuse the functions dealing with X to save time of
        # programming :)
        y = y.reshape(-1, 1)
        feat2index_dict_y = raw2index(y)
        apply_raw2index(y, feat2index_dict_y)
        feat2index_dict_y = feat2index_dict_y[0]

        # compute prior probility
        y = y.reshape(-1)

        prior = np.zeros((len(feat2index_dict_y),))
        y_count = np.zeros((len(feat2index_dict_y),))
        for i in y:
            y_count[i] +=1
        prior = y_count / np.sum(y_count)

        # compute the marginal probility P(X_i=x| Y=y)
        conditionals = [
            [
                np.zeros((len(feat2index_dict_x[i]), ))
                for i in range(len(feat2index_dict_x))
            ]
            for _ in range(len(feat2index_dict_y))
        ]
        # Note: conditionals.shape =
        # (output_val_index, input_feat_index, input_feat_val_index)

        # Compute marginal probilibty by maximum likelihood estimation
        for i in range(X.shape[0]):
            now_x = X[i]
            now_y = y[i]
            for j in range(X.shape[1]):
                conditionals[now_y][j][now_x[j]] += 1 / y_count[now_y]

        # Save neccesaty information
        # (If you are not clear for any of these, be active to query them
        # by the function generated them)
        self.feat2index_dict_x = feat2index_dict_x
        self.feat2index_dict_y = feat2index_dict_y
        self.prior = prior
        self.conditionals = conditionals
        self.d = d

    def predict(self, X):
        """
        Predict the result after trainning
        :param X: numpy array, a design matrix, with shape (num_input x dimension_feature)
        :return: (pred, prob, input_prob)
        pred: a 1-d np-array, of which each element represents the predicted result of
        the input designed matrix (with the same order) (e.g: ['yes', 'no', 'no', ...])
        prob: the posterior probability of each prediction (P(Y=y|X_1=x_1, X_2=x_2, ..))
        input_prob: the marginal probability of each input
         (P(X_1=x_1, X_2=x_2, ..))
        """
        X = copy.deepcopy(X)
        self.d.apply(X)
        apply_raw2index(X, self.feat2index_dict_x)
        y = []
        prob = []
        input_prob = []
        for i in range(X.shape[0]):
            now_x = X[i]
            joint_prob = copy.deepcopy(self.prior)
            for j in range(X.shape[1]):
                for k in range(len(self.feat2index_dict_y)):
                    joint_prob[k] = joint_prob[k] * self.conditionals[k][j][now_x[j]]

            pred = np.argmax(joint_prob)
            y.append(pred)
            prob.append(joint_prob[pred] / np.sum(joint_prob))
            input_prob.append(np.sum(joint_prob))

        return np.array(apply_index2raw(y, self.feat2index_dict_y)), prob, input_prob



if __name__ == "__main__":
    p_x_train = os.path.join("data", "x_train.pkl")
    p_y_train = os.path.join("data", "y_train.pkl")

    p_x_test = os.path.join("data", "x_test.pkl")
    p_y_test = os.path.join("data", "y_test.pkl")

    X_train = load(p_x_train)
    y_train = load(p_y_train)
    X_test = load(p_x_test)
    y_test = load(p_y_test)


    model = Naive_bayes()
    # we are going to split dimension of age, BMI and charges
    model.fit(X_train, y_train, discretize_dim=(0, 2, 5), split_num=(5, 5, 5))

    pred, _, _ = model.predict(X_train)
    train_acc = np.sum((pred == y_train)) / y_train.shape[0]

    pred, _, _ = model.predict(X_test)
    test_acc = np.sum((pred == y_test)) / y_test.shape[0]
    print("train accuracy: {}, test accuracy: {}".format(train_acc, test_acc))
