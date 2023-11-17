# Python, using Anaconda environment
# Week 2, Day 7
import numpy as np
from sklearn.naive_bayes import BernoulliNB

class MyBayes:
    def __init__(self):
        self.training_size = 0
        self.num_of_variable = 0
        self.PXI = None
        self.PI = None
        self.correct = None
        self.incorrect = None

    def fit(self, XTrain, YTrain):
        self.PXI = np.zeros((2, XTrain.shape[1], 2))
        self.PI = np.zeros(2)
        # print(self.PXI)
        # print(self.PI)
        self.training_size = XTrain.shape[0]
        self.num_of_variable = XTrain.shape[1]
        for k in range(XTrain.shape[1]):
            self.PXI[1, k, 1] = np.sum(np.logical_and(XTrain[:, k], YTrain))
            self.PXI[1, k, 0] = np.sum(np.logical_and(np.logical_not(XTrain[:, k]), YTrain))
            self.PXI[0, k, 1] = np.sum(np.logical_and(XTrain[:, k], np.logical_not(YTrain)))
            self.PXI[0, k, 0] = np.sum(np.logical_not(np.logical_or(XTrain[:, k], YTrain)))
        self.PI[1] = np.sum(YTrain)
        self.PI[0] = self.training_size  - self.PI[1]
        # print(self.PXI)
        # print(self.PI)
        self.PXI = (self.PXI + 1 / 2) / (self.training_size  + 1)  # anti zero division
        self.PI = self.PI / self.training_size
        # print(self.PXI)
        # print(self.PI)

        return self

    def predict1(self, x):
        try:
            x = x.astype(int)
        except:
            x = int(x)
        P = np.zeros_like(self.PI)
        allofthem = np.arange(self.num_of_variable)
        # print(P)
        # print(allofthem)

        for ii in range(len(self.PI)):
            P[ii] = np.prod(self.PXI[ii, allofthem, x]) * self.PI[ii]
        # print(P)
        denominator = np.sum(P)
        P = P / denominator
        # print(P)

        choosenClass = np.argmax(P)
        return choosenClass

    def score(self, XTest, YTest):
        self.correct = np.zeros(2)
        self.incorrect = np.zeros(2)

        for ii in range(XTest.shape[0]):
            klasse = self.predict1(XTest[ii, :].astype(int))
            if klasse == YTest[ii]:
                self.correct[klasse] = self.correct[klasse] + 1
            else:
                self.incorrect[klasse] = self.incorrect[klasse] + 1
        print(f"Von {XTest.shape[0]} Testfaellen wurden {np.sum(self.correct)} richtig und {np.sum(self.incorrect)} falsch klassifiziert")

        return self

    def predict2(self):
        ####
        pass




def test_train_split(X, y_target):
    allData = np.arange(0, X.shape[0])
    iTesting = np.random.choice(X.shape[0], int(X.shape[0] * 0.2), replace=False)
    iTraining = np.delete(allData, iTesting)
    # dataRecords = len(iTraining)
    XTrain = X[iTraining, :]
    YTrain = y_target[iTraining]
    XTest = X[iTesting, :]
    YTest = y_target[iTesting]

    return XTrain, XTest, YTrain, YTest









# end of file
