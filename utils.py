#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: Yin-tao Xu
# @Date  : 18-9-8
# @Desc  : This script mainly aims at providing user-friendly interface for
# serializing basing on pickle.

import pickle

def load(path):
    """
    load the pickle-serilized python object
    :param path: the path you want to load serialized python object
    :return: the target python object
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def save(obj, path):
    """
    do the serialization by pickle
    :param obj: target python object
    :param path: the path you want to save the serialized python object
    :return: None
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
