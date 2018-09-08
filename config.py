#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: Yin-tao Xu
# @Date  : 18-9-8
# @Desc  : configuration for this project

# Note: To let the programme less dependcy on the 3rd library,
#       author removes the usage of EasyDict here. However, by convention,
#       EasyDict is frequently used as the configuration of the program

cfg = {}
# the desciption of each feature
cfg['desc'] = ['age' , 'sex' , 'bmi' , 'children', 'region', 'charges']
# the description of output
cfg['output_desc'] = 'smoke?'

