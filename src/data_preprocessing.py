# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:33:04 2026

@author: 0125i
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def select_top_features(X_train, y_train, X_test, top_n=16, random_state=26):
    """
    利用隨機森林的 Gini Importance 篩選出最具影響力的 Top N 特徵
    """
    print(f"正在執行特徵篩選 (Top {top_n} features)...")
    rf_base = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    
    importances = rf_base.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    top_features = imp_df['Feature'].head(top_n).tolist()
    print(f"✅ 已篩選出核心特徵：{top_features}")
    
    # 回傳切片後的資料
    return X_train[top_features], X_test[top_features], top_features

def standardize_data(X_train, X_test):
    """
    進行 Z-score 標準化，防止神經網路與 SVM 發生梯度消失或權重偏移
    """
    print("正在執行資料標準化 (Standardization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # 測試集只能用訓練集的標準轉換
    
    return X_train_scaled, X_test_scaled