# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:33:52 2026

@author: 0125i
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow.keras.backend as K

# ==========================================
# 決策樹 & 隨機森林專用 Threshold 包裝盒
# ==========================================
class ThresholdRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=1000, max_features=3, random_state=26, threshold=0.5):
        self.n_estimators = n_estimators 
        self.max_features = max_features  
        self.random_state = random_state
        self.threshold = threshold
        self.model_ = None

    def fit(self, X, y):
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            max_features=self.max_features, 
            random_state=self.random_state,
            n_jobs=-1 
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        probs = self.model_.predict_proba(X)
        return np.where(probs[:, 0] > self.threshold, 0, 1)
        
    def predict_proba(self, X):
        return self.model_.predict_proba(X)

# ==========================================
# 神經網路 (Keras) 專用 CV 包裝盒與函數
# ==========================================
def macro_f1(y_true, y_pred):
    """Keras 專用的自訂 Macro F1 評估指標"""
    y_pred_labels = K.round(y_pred)
    y_true = K.cast(y_true, 'float32')

    tp = K.sum(y_true * y_pred_labels)
    tn = K.sum((1 - y_true) * (1 - y_pred_labels))
    fp = K.sum((1 - y_true) * y_pred_labels)
    fn = K.sum(y_true * (1 - y_pred_labels))

    p_1 = tp / (tp + fp + K.epsilon())
    r_1 = tp / (tp + fn + K.epsilon())
    f1_1 = 2 * p_1 * r_1 / (p_1 + r_1 + K.epsilon())

    p_0 = tn / (tn + fn + K.epsilon())
    r_0 = tn / (tn + fp + K.epsilon())
    f1_0 = 2 * p_0 * r_0 / (p_0 + r_0 + K.epsilon())

    return (f1_0 + f1_1) / 2.0

def build_new_nn_model(input_dim):
    """產出全新、權重歸零的神經網路工廠函數"""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[macro_f1])
    return model

class KerasCVWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=50, batch_size=64, threshold=0.5):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_ = None

    def fit(self, X, y):
        self.model_ = build_new_nn_model(self.input_dim)
        # 加入 1:5 的非對稱成本權重
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, 
                        class_weight={0: 1.0, 1: 5.0}, verbose=0)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        probs = self.model_.predict(X, verbose=0).flatten()
        return np.where(probs > self.threshold, 1, 0)
        
    def predict_proba(self, X):
        probs = self.model_.predict(X, verbose=0).flatten()
        return np.column_stack((1 - probs, probs))