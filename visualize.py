#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : visualize.py
# @Author: Yin-tao Xu
# @Date  : 18-9-8
# @Desc  :

from model import Naive_bayes, apply_raw2index
from config import cfg
import os
import numpy as np
from utils import load
import copy
import matplotlib.pyplot as plt
import math

def visualize_joint_prob(model, obj_feat_dim, cfg, x=None, ax=None, pred=-1):
    joint_dist = []
    for output_t in range(len(model.feat2index_dict_y)):
        joint_dist.append(model.conditionals[output_t][obj_feat_dim] * model.prior[output_t])
    joint_dist = np.array(joint_dist)
    y_feat_raw = list(model.feat2index_dict_y.keys())
    x_feat_raw = None
    if obj_feat_dim in model.d.todo_index:
        x_feat_raw = []
        d_index = model.d.todo_index.index(obj_feat_dim)
        prev_str = '-∞'
        interval = model.d.intervals[d_index]
        prev_val = model.d.feat_mins[d_index]
        for _ in range(model.d.split_num[d_index] - 1):
            now_val = prev_val + interval
            now_str = "{:.2f}".format(now_val)
            x_feat_raw.append(r"{} ~ {}".format(prev_str, now_str))
            prev_val = now_val
            prev_str = now_str
        x_feat_raw.append(r"{:.2f} ~ +∞".format(prev_val))
    else:
        x_feat_raw = list(model.feat2index_dict_x[obj_feat_dim].keys())

    plot_show = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plot_show = True
    plt.imshow(joint_dist, cmap='Blues')
    ax.set_yticks([_ for _ in range(joint_dist.shape[0])])
    ax.set_yticklabels(y_feat_raw)
    ax.set_xticks([_ for _ in range(joint_dist.shape[1])])
    ax.set_xticklabels(x_feat_raw, rotation=45)
    ax.set_xlabel(cfg['desc'][obj_feat_dim], fontsize='x-large')
    ax.set_ylabel(cfg['output_desc'], fontsize='x-large')

    raw_x = x
    x = np.array([copy.deepcopy(x)])
    model.d.apply(x)
    apply_raw2index(x, model.feat2index_dict_x)
    x = x[0]

    m_p = []
    if x is not None:
        for pos_y in range(len(model.feat2index_dict_y)):
            text_content = model.conditionals[pos_y]
            text_content = text_content[obj_feat_dim]
            feat_val_index = x[obj_feat_dim]
            text_content = text_content[feat_val_index]
            m_p.append(text_content)
        for pos_y in range(len(model.feat2index_dict_y)):
            # output_type = ['yes', 'no']
            # text_content = "P({}={}|Y={})\n{:.2f}".format(cfg['desc'][obj_feat_dim],
            #                                               raw_x[obj_feat_dim],
            #                                            output_type[pos_y],
            #                                            m_p[pos_y])
            text_content = "P(X|Y)={:.4f}".format(m_p[pos_y])

            if pos_y != pred:
                bbox_props = dict(boxstyle="round4, pad=0.3", fc="white", ec="gray", lw=1)
                ax.text(feat_val_index, pos_y, text_content, ha="center", va="center", rotation=0,
                            size=10,
                            bbox=bbox_props)
            else:
                bbox_props = dict(boxstyle="round4, pad=0.3", fc="white", ec="black", lw=2)
                ax.text(feat_val_index, pos_y, text_content, ha="center", va="center", rotation=0,
                            size=10,
                            bbox=bbox_props)
    plt.title('P(smoke, {})'.format(cfg['desc'][obj_feat_dim]), fontsize=20)
    plt.colorbar()
    if plot_show:
        plt.show()

def plot_computation_process(model, x):
    feat_num = x.shape[0]
    pred, prob, input_prob = model.predict(np.array([x]))
    pred_index = model.feat2index_dict_y[pred[0]]
    print("Prediction result: {} | Posterior probability: {:.4f}".format(pred[0], prob[0]))

    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    fig = plt.figure()
    print("P(Y='yes')={:.4f}, P(Y='no')={:.4f}".format(model.prior[model.feat2index_dict_y['yes']],
                                                       model.prior[model.feat2index_dict_y['no']]))
    header = None
    for i in range(feat_num):
        if header is None:
            header = "P({}={}".format(cfg['desc'][i], x[i])
        else:
            header += ", {}={}".format(cfg['desc'][i], x[i])
    print(header + ")={:4f}".format(input_prob[0]))
    print("You can check the posterior probability by yourself!")
    print("Marginal probability are in the figures.")

    for feat_index in range(feat_num):
        ax = fig.add_subplot(math.ceil(float(feat_num + 1) / 3),  3, feat_index + 1)
        visualize_joint_prob(model, feat_index, cfg, x=x, ax=ax, pred=pred_index)

    plt.subplots_adjust(wspace=0.3, hspace=1.5)
    plt.show()

def output_basic_info(x):
    print("=========info==========")
    for i in range(len(x)):
        print(cfg["desc"][i], end="")
        print("=", end="")
        print(x[i])
    print("=========================")

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

    for i in range(10, 20):
        output_basic_info(X_test[i])
        plot_computation_process(model, X_test[i])
