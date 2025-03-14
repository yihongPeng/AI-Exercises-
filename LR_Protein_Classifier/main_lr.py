import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt


class LRModel:
    # todo:09
    """
        Initialize Logistic Regression (from sklearn) model.

    """
    def __init__(self):
        self.model = LogisticRegression(C=10,penalty='l2',solver='saga',max_iter=1000,tol=1e-3)

    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """
    def train(self,train_data,train_targets):
        self.model.fit(train_data,train_targets)
        return
    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """
    def evaluate(self, data, targets):
        return self.model.score(data,targets)


class LRFromScratch:
    # todo:
    def __init__(self):
        self.W = np.ones(shape=(300,))
        self.b = np.zeros(1)
        self.lr = 0.1
        self.alpha = 0.1
        self.max_iters = 1000

    def train(self, train_data, train_targets):
        for _ in range(self.max_iters):
            z = np.matmul(train_data,self.W) +self.b
            y_hat = 1 / (1 + np.exp(-z))
            # loss = np.mean((- train_targets * np.log(y_hat)))
            self.W -= (self.W * np.mean(y_hat - train_targets) + 2 * self.alpha * self.W) * self.lr
            self.b -= (np.mean(y_hat - train_targets) + 2 * self.alpha * self.b) * self.lr


    def evaluate(self, data, targets):
        z = np.matmul(data,self.W) +self.b
        y_hat = 1 / (1 + np.exp(-z))
        y_hat = np.int16(y_hat >= 0.5)
        return np.mean(y_hat == targets)


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    print(diagrams.shape)
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'
    # print(cast.columns.values,cast.shape,diagrams.shape,type(diagrams))
    data_list = []
    target_list = []

    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task].values
        
        train_data = diagrams[(task_col == 1) + (task_col == 2)]
        train_targets = task_col[(task_col == 1) + (task_col == 2)]
        test_data = diagrams[(task_col == 3) + (task_col == 4)]
        test_targets = task_col[(task_col == 3) + (task_col == 4)]
        train_targets[train_targets == 2] = 0
        test_targets[test_targets == 3] = 1
        test_targets[test_targets == 4] = 0
        # print(train_data.shape,train_targets.shape)
        ## todo: Try to load data/target
        # train_data,test_data,train_targets,test_targets = train_test_split(diagrams,task_col.values,train_size=0.8,random_state=42,shuffle=True)

        # print(train_data.shape,test_data.shape,train_targets.shape,test_targets.shape)
        # print(train_targets)
        # assert 1<0,"break here"
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))

    return data_list, target_list

def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []
    

    # model = LRModel()
    model = LRFromScratch()

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)
        


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))
    # 画柱状图
    task_indices = np.arange(len(task_acc_train))  # 任务编号
    bar_width = 0.4  # 柱子的宽度

    plt.figure(figsize=(12, 6))
    plt.bar(task_indices - bar_width / 2, task_acc_train, bar_width, label="Train Accuracy", color='b', alpha=0.7)
    plt.bar(task_indices + bar_width / 2, task_acc_test, bar_width, label="Test Accuracy", color='r', alpha=0.7)

    plt.xlabel("Task Index")
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression Model Accuracy per Task")
    plt.xticks(task_indices)  # 使 x 轴刻度与任务编号对齐
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)

