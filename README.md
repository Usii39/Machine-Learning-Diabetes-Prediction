# 🩸 糖尿病風險預測與機器學習模型優化 (Diabetes Risk Prediction)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg)

## 📌 專案簡介 (Project Overview)
本專案旨在利用機器學習演算法，針對包含高度不平衡特徵的醫療數據進行「糖尿病風險預測」。
有別於一般僅呼叫 API 的初階模型實作，本專案深入演算法底層，解決了**資料不平衡 (Imbalanced Data)**、**跨框架交叉驗證 (Cross-Validation Wrapper)** 以及**決策邊界尋優 (Threshold Tuning)** 等實務痛點，旨在建立具備高泛化能力與高解釋性的醫療預測模型。

## 💡 核心技術亮點 (Key Methodologies)
- **非對稱成本敏感學習 (Cost-Sensitive Learning)**：針對醫療場域「漏判糖尿病」的高昂代價，全面導入 `class_weight` (1:5) 機制，強迫模型關注少數類別。
- <img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/77174075-938d-4b11-8bb4-4f2610e2b887" />

- **動態特徵篩選 (Feature Selection)**：利用隨機森林的 Gini Importance，精準萃取出前 16 項核心特徵（保留 >90% 資訊量），有效降低維度災難並加速後續矩陣運算。
- **高階模型封裝 (Custom Estimator Wrappers)**：透過繼承 `BaseEstimator`，將神經網路 (Keras) 與自訂的決策邊界邏輯封裝為標準 scikit-learn API，完美對接 `GridSearchCV` 進行自動化超參數尋優，杜絕權重洩漏 (Weight Leakage)。
- **事後尋優法 (Post-Training Threshold Tuning)**：揚棄傳統 0.5 的預設機率切分，在獨立驗證集上透過陣列運算光速掃描 80 種 Threshold，極大化 Macro F1 分數。

<img width="1383" height="983" alt="image" src="https://github.com/user-attachments/assets/61477e35-71c8-43bd-89d6-f5e706e99ec4" />
<img width="668" height="583" alt="image" src="https://github.com/user-attachments/assets/1223c1b5-524c-4f74-96b3-8fa72c974fb0" />



## 🛠️ 實作模型與成效 (Models & Performance)
本專案統一採用 **Macro F1 Score** 作為核心評估指標，以公允衡量模型在少數類別上的表現。

| 機器學習演算法 | 核心優化策略 | 測試集 Macro F1 |
| :--- | :--- | :---: |
| **Decision Tree** | Cost-Complexity Pruning (CCP), 均勻抽樣 Alpha 剪枝 | `0.6167` |
| **Random Forest** | Gini 特徵篩選 (Top 16), 雙層 Grid Search (`mtry` & `threshold`) | `0.6636` |
| **Neural Network (MLP)** | Standardization, Dropout, 自訂 Macro F1 評估器, 早停機制 (Early Stopping) | `0.6804` |
| **Support Vector Machine** | RBF Kernel, Platt Scaling 機率校正, 超參數尋優 | ` 0.6298` |
| **XGBoost** |  | `0.67769` |
