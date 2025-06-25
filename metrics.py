#!/usr/bin/env python
# coding: utf-8

# This file provides a set of utility functions to evaluate the performance of binary classification models.
# 
# Implemented metrics include:
# - **Precision:** Out of all predictions for this class, how many were correct
#       **Formula:** TP / (TP + FP)
# - **Recall:** Out of all actual examples of this class, how many did we correctly predict
#             **Formula: TP / (TP + FN)
# - **F1-score:** Harmonic mean of precision and recall. A balanced metric.
#           **Formula:** 2* (precision * recall) / (precision + recall)
# - **Accuracy:** Overall correct predictions out of all predictions.
#          **Formula:** (TP + TN) / Total
# 
# All functions assume that labels are in {-1, +1} format.
# 
# In addition, plotting utilities are provided to visualize:
# - Training loss over iterations
# - metrics over iterations
# 
# These tools are model-agnostic and can be reused for both Support Vector Machines (SVM) and Logistic Regression implementations or other algorithms.
# 

# In[2]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[1]:


def precision(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1)) # True positive
    fp = np.sum((y_true == -1) & (y_pred == 1)) # False positive
    denom = tp + fp
    
    if denom == 0:
        return 0.0 
    return tp / denom
    
def recall(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positive
    fn = np.sum((y_true == 1) & (y_pred == -1)) # False negative
    denom = tp + fn
    
    if denom == 0:
        return 0.0 
    return tp / denom

def f1_score(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    denom = prec + recal
    
    if denom == 0:
        return 0.0
    return 2 * prec * recal / denom
    
def accuracy(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tp = np.sum((y_true == 1) & (y_pred == 1)) # True positive
    tn = np.sum((y_true == -1) & (y_pred == -1)) # True negative
    return (tp + tn)/ len(y_true)


# The following functions displays the evolution of a metric (accuracy, precision, recall, f1 score) over the iterations.
# 
# - list_iter: List of iterations.
# - list_predictions: List of predictions made at each iteration.
# - metric_fct : Metric function used
# - y_true : The real labels in the training database.

# In[1]:


def plot_loss(list_iter, list_loss):
    plt.plot(list_iter, list_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
    

def plot_metrics(list_iter, list_metrics, metric_fct):
    plt.plot(list_iter, list_metrics)
    plt.xlabel("Iteration")
    plt.ylabel(f"Training {metric_fct.__name__}")
    plt.title(f"Training {metric_fct.__name__} Curve")
    plt.grid(True)
    plt.show()
    
    
'''def plot_metrics(list_iter, list_predictions, metric_fct, y_true):
    list_metric = []
    for y_pred in list_predictions:
        score = metric_fct(y_true, y_pred)
        list_metric.append(score)

    plt.plot(list_iter, list_metric)
    plt.xlabel("Iteration")
    plt.ylabel(f"Training {metric_fct.__name__}")
    plt.title(f"Training {metric_fct.__name__} Curve")
    plt.grid(True)
    plt.show()'''


# In[ ]:




