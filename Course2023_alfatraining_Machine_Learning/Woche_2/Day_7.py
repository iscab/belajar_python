# Python, using Anaconda environment
# Week 2, Day 7
import numpy as np
import helper_func.day7func as hfun


myBayes = hfun.MyBayes()

# Datenerzeugen
X = np.random.randint(0, 2, size=(20,2))
#print(X)
y = np.logical_and(X[:,0], X[:,1])
#print(y)

# test and train data split
XTrain, XTest, YTrain, YTest = hfun.test_train_split(X, y)
# print(XTrain.shape)
# print(XTest.shape)
# print(YTrain.shape)
# print(YTest.shape)
# print("\n")

myBayes = myBayes.fit(XTrain, YTrain)
# print(myBayes.PXI)
# print(myBayes.PI)
print("\n")

theclass = myBayes.predict1(0)
print(theclass)

myBayes = myBayes.score(XTest, YTest)
print(myBayes.correct)
print(myBayes.incorrect)
print("\n")

# real data
print("real data")
dataset = np.loadtxt("Bayes\\diagnosis.csv", delimiter=",")
X = dataset[:,1:6]
Y = dataset[:,7]

XTrain, XTest, YTrain, YTest = hfun.test_train_split(X, Y)
myBayes = myBayes.fit(XTrain, YTrain)
myBayes = myBayes.score(XTest, YTest)
print(myBayes.correct)
print(myBayes.incorrect)
print("\n")


# end of file
