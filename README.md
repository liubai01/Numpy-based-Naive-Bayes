# Numpy-based-Naive-Bayes
### Introduction to the repository

It is a case study of naive Bayes estimation when I read a textbook about statical learning, 统计学习方法, 李航(Introduction to statistical learning methods, Hang Li). A good method to study a new model is always to get over the whole process by yourself without the help of 3-rd open source package(like sci-kit learn). 

This repository aims at providing reader a more practical and intuitive way to learn naive Bayes, by providing the visualization of joint probability. 

![](https://github.com/liubai01/Numpy-based-Naive-Bayes/img/joint_distribution.png)

The implementation bases the task of predicting whether a person is smoke or not. The dataset is provided by the book 'Machine Learning with R' on [Kaggle](https://www.kaggle.com/mirichoi0218/insurance).  My implementation shows the test accuracy 0.92 and shows the main factor of a person smoke or not is the individual medical costs.

### Introduction to dataset&task

All we get is 1,338 entires.  You can get more detailed information on [Kaggle](https://www.kaggle.com/mirichoi0218/insurance).

**Columns**:

- age: age of primary beneficiary 
- sex: insurance contractor gender, female, male 
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,         objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9 
- children: Number of children covered by health insurance / Number of dependents
- smoker: Smoking
- region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
- charges: Individual medical costs billed by health insurance

The recommended task on Kaggle is to estimate the individual medical costs. [One popular kernel](https://www.kaggle.com/grosvenpaul/regression-eda-and-statistics-tutorial) shows that smoke or not is an important feature to estimate the cost. Then, here is my question:

Is it possible to estimate a person is a smoker or not with the help of other variables?

### Prerequisite

1. Python 3
2. [Scikit-Learn](http://scikit-learn.org/stable/documentation.html)(we only use to partition the original dataset into training and test set)
3. [Numpy](http://www.numpy.org/)(know some basic operators)
4. Matplotlib

Recommendation:  [Anaconda](https://www.anaconda.com/download/) is a one-stage solution instead of manually installing these 3-rd party package.

### Quick start

1. Open terminal at the directory you want to clone the repository.

```shell
git clone https://github.com/liubai01/Numpy-based-Naive-Bayes.git
cd Numpy-based-Naive-Bayes/
```

2. Download dataset from [Kaggle](https://www.kaggle.com/mirichoi0218/insurance), extract the `insurance.csv` to the path `./data/`
3. Run the demo.

```shell
python main.py 
```

4. You will see training accuracy, test accuracy and some detailed information of a sample. To learn about how naive Bayes works, I suggest you to manually compute the posterior probability by the information and a figure given by the demo.

```
train accuracy: 0.9271028037383178, test accuracy: 0.9291044776119403
=========info==========
age=38
sex=female
bmi=30.21
children=3
region=northwest
charges=7537.1639
=========================
Prediction result: no | Posterior probability: 0.9984
...
```

### Work-flow

A brief summary of work-flow of this case study.

1. Load in the data, and split it into training set and test set.
2. Feature engineering: discretize continuous input random variables.
3. Train the naive Bayes classifier
4. Test and see the result!

Tree structure of critical files in repository and corresponding steps(mentioned above).

```
├── config.py
├── data
│   └── insurance.csv (dataset file)
├── data_spliter.py(Step 1)
├── main.py(entry of the program)
├── model.py(Step 2, 3)
├── utils.py(tools of serialization)
└── visualize.py(Step 4)
```

### Summary

Any suggestion or advice is welcome. A Jupyternote book will be on-line recently(in Chinese).