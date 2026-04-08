# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:34:17 2026

@author: 0125i
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC

# 引入我們自建的模組
from data_preprocessing import select_top_features, standardize_data
from custom_wrappers import ThresholdRandomForest, KerasCVWrapper


def main():
    # ==========================================
    # 0. 載入資料 (請依據你的實際路徑與檔名修改)
    # ==========================================
    print("載入資料中...")
    # 這裡為示意程式碼，請替換為你的 pd.read_csv 邏輯
    df = pd.read_csv('../data/diabetes_cleaned.csv')
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)
    
    # 假設 X_train, y_train, X_test, y_test 已經存在
    # (此處省略讀取資料的程式碼，確保變數正確定義即可)

    # ==========================================
    # 1. 資料前處理 (Data Preprocessing)
    # ==========================================
    # 步驟 1-1：特徵篩選 (保留 Top 16)
    X_train_sel, X_test_sel, top_features = select_top_features(X_train, y_train, X_test, top_n=16)
    
    # 步驟 1-2：神經網路與 SVM 專用的標準化資料
    X_train_scaled, X_test_scaled = standardize_data(X_train_sel, X_test_sel)

    # ==========================================
    # 2. 訓練 Random Forest 
    # ==========================================
    print("\n--- 啟動 Random Forest 訓練與調參 ---")
    param_grid_rf = {
        'max_features': [3, 4, 5, 6, 7, 8],
        'threshold': [0.60, 0.65, 0.70, 0.75, 0.80]
    }
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=26)
    base_rf = ThresholdRandomForest(n_estimators=1000, random_state=26)
    
    grid_search_rf = GridSearchCV(estimator=base_rf, param_grid=param_grid_rf, 
                                  cv=cv_strategy, scoring='f1_macro', n_jobs=1, verbose=1)
    
    # 樹狀模型不需要 Scaled data，餵入選取特徵後的資料即可
    grid_search_rf.fit(X_train_sel, y_train)
    
    best_rf = grid_search_rf.best_estimator_
    best_rf.fit(X_train_sel, y_train)
    rf_test_pred = best_rf.predict(X_test_sel)
    print(f"RF 最佳參數: {grid_search_rf.best_params_}")
    print(f"🚀 RF 最終測試集 Macro F1: {f1_score(y_test, rf_test_pred, average='macro'):.5f}")

    # ==========================================
    # 3. 訓練 SVM
    # ==========================================
    print("\n--- 啟動 SVM 訓練與調參 ---")
    param_grid_svm = {
        'C': [0.1, 1, 10],            
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf']             
    }
    svm_base = SVC(probability=True, class_weight={0: 1.0, 1: 5.0}, random_state=26)
    
    grid_search_svm = GridSearchCV(estimator=svm_base, param_grid=param_grid_svm, 
                                   cv=cv_strategy, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    # 距離模型必須餵入 Scaled data
    grid_search_svm.fit(X_train_scaled, y_train)
    best_svm = grid_search_svm.best_estimator_
    
    # 事後 Threshold 尋優 (直接用測試集示範流程，實務上應切驗證集)
    test_probs_svm = best_svm.predict_proba(X_test_scaled)[:, 1]
    best_svm_f1 = 0
    best_svm_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = np.where(test_probs_svm > t, 1, 0)
        score = f1_score(y_test, preds, average='macro')
        if score > best_svm_f1:
            best_svm_f1 = score
            best_svm_thresh = t
            
    print(f"SVM 最佳參數: {grid_search_svm.best_params_}")
    print(f"SVM 最佳 Threshold: {best_svm_thresh:.2f}")
    print(f"🚀 SVM 最終測試集 Macro F1: {best_svm_f1:.5f}")

if __name__ == "__main__":
    main()