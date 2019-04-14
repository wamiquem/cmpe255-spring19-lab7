import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 

# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 

    # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  

    # train
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    return  X_train, X_test, y_train, y_test

def polynomial_kernel(X_train, X_test, y_train, y_test):
    svc_polykernel = SVC(kernel='poly', degree=8)  
    svc_polykernel.fit(X_train, y_train)

    y_pred = svc_polykernel.predict(X_test) 
    
    print('\n','***********Polynomial kernel***********')
    print('CONFUSION MATRIX:')
    print(confusion_matrix(y_test, y_pred))  
    print('\n','CLASSIFICATION REPORT:')
    print(classification_report(y_test, y_pred)) 

def gaussian_kernel(X_train, X_test, y_train, y_test):
    svc_gauskernel = SVC(kernel='rbf')  
    svc_gauskernel.fit(X_train, y_train)

    y_pred = svc_gauskernel.predict(X_test) 

    print('\n','***********Gaussian kernel***********','\n')
    print('CONFUSION MATRIX:')
    print(confusion_matrix(y_test, y_pred))  
    print('\n','CLASSIFICATION REPORT:')
    print(classification_report(y_test, y_pred)) 

def sigmoid_kernel(X_train, X_test, y_train, y_test):
    svc_sigkernel = SVC(kernel='sigmoid')  
    svc_sigkernel.fit(X_train, y_train)

    y_pred = svc_sigkernel.predict(X_test) 

    print('\n','***********Sigmoid kernel***********')
    print('CONFUSION MATRIX:')
    print(confusion_matrix(y_test, y_pred))  
    print('\n','CLASSIFICATION REPORT:')
    print(classification_report(y_test, y_pred)) 

def test():
    X_train, X_test, y_train, y_test = import_iris()
    polynomial_kernel(X_train, X_test, y_train, y_test)
    gaussian_kernel(X_train, X_test, y_train, y_test)
    sigmoid_kernel(X_train, X_test, y_train, y_test)

test()