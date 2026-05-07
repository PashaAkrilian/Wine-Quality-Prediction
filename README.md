# Omni-Signal Residual DCN for Wine Quality Classification

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Scikit--Learn-green.svg)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-red.svg)

A deep learning and feature engineering project for **Wine Quality Classification** using an advanced architecture called **Omni-Signal Residual Deep & Cross Network (DCN)**.

This notebook combines:

* enological feature engineering
* latent clustering
* residual deep learning
* cross feature interaction modeling
* ordinal-aware reasoning
* imbalance handling
* soft-label knowledge distillation

The goal is to maximize multiclass wine quality prediction performance by capturing both explicit chemical relationships and latent structural patterns within physicochemical wine data.

---

# Project Overview

Wine quality prediction is a challenging multiclass classification task because wine characteristics often contain:

* nonlinear interactions
* noisy measurements
* class imbalance
* hidden latent groups
* ordinal quality relationships

This project introduces a robust hybrid pipeline combining:

* **Enological Feature Engineering**
* **Residual Deep Neural Networks**
* **Deep & Cross Network (DCN)**
* **Latent Clustering using K-Means**
* **Mutual Information + VIF diagnostics**
* **SMOTE + Tomek Links balancing**
* **Knowledge Distillation with soft labels**

The final architecture is designed to improve:

* multiclass separation
* feature interaction learning
* ordinal sensitivity
* robustness against noisy observations

---

# Repository Structure

```bash
Omni-Signal-Residual-DCN/
│
├── omni-signal-residual-dcn-advanced-ordinal-reasoni.ipynb
├── outputs/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   └── submission.csv
├── LICENSE
└── README.md
```

---

# Workflow

```text
Load Wine Dataset
        ↓
Enological Feature Engineering
        ↓
Latent Clustering using KMeans
        ↓
Signal & Redundancy Diagnostics (MI + VIF)
        ↓
Advanced Feature Transformation
        ↓
Outlier Clipping & Normalization
        ↓
Knowledge Distillation Features
        ↓
SMOTE + Tomek Links Balancing
        ↓
Residual Deep & Cross Network Training
        ↓
Evaluation & Prediction
```

---

# Methodology

# 1. Enological Feature Engineering

The notebook creates domain-driven wine chemistry features to improve predictive signal quality.

Examples include:

* sulfur management indicators
* acidity interaction features
* alcohol-sulphates interaction
* sulfur efficiency ratios
* volatile acidity ratios
* transformed sugar & sulfur variables

These engineered features help the model better capture wine composition behavior.

---

# 2. Latent Clustering

The project applies:

```python
KMeans(n_clusters=4)
```

to discover hidden wine-type structures based on:

* alcohol
* volatile acidity
* sulfur dioxide
* pH

The resulting cluster labels are injected as additional predictive features.

This allows the network to learn latent wine behavior patterns not directly visible from raw variables.

---

# 3. Feature Diagnostics

To evaluate feature quality, the notebook computes:

## Mutual Information (MI)

Measures feature predictive signal strength.

## Variance Inflation Factor (VIF)

Measures multicollinearity and redundancy.

This dual diagnostic system helps retain strong features while minimizing unstable correlations.

---

# 4. Advanced Feature Transformation

The preprocessing stage includes:

* logarithmic transformation
* ratio injection
* nonlinear interaction synthesis
* skewness reduction
* distribution stabilization

Examples:

```python
np.log1p()
```

for sulfur and sugar related variables.

---

# 5. Robust Outlier Handling

Extreme values are clipped using:

```text
0.5% – 99.5% percentile clipping
```

This preserves information while reducing instability caused by heavy-tailed distributions.

---

# 6. Knowledge Distillation & Soft Labels

The notebook introduces:

* out-of-fold prediction features
* probabilistic soft labels
* distilled meta-information

This allows the final network to leverage auxiliary predictive structure learned during cross-validation.

---

# 7. Imbalanced Data Handling

Wine quality labels are imbalanced, especially for rare quality classes.

The project uses:

```text
SMOTE + Tomek Links
```

with customized nearest-neighbor configuration to:

* oversample minority classes
* remove overlapping samples
* improve class separation

---

# 8. Residual Deep & Cross Network (DCN)

The final predictive engine combines:

* residual dense blocks
* cross feature interaction layers
* ordinal-aware reasoning
* normalization layers
* dropout regularization

This architecture is designed to:

* learn high-order feature interactions
* stabilize gradient flow
* improve multiclass discrimination
* preserve ordinal quality relationships

---

# Example Usage

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

### Neural Network Training

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=128
)
```

---

# Tech Stack

* Python
* NumPy
* pandas
* TensorFlow / Keras
* scikit-learn
* imbalanced-learn
* Optuna
* seaborn
* matplotlib

---

# Key Outputs

The notebook generates:

* feature diagnostic plots
* cluster visualization
* class distribution analysis
* training history curves
* confusion matrix
* validation metrics
* multiclass predictions

---

# Use Cases

This project is highly relevant for:

* wine quality analytics
* multiclass classification research
* tabular deep learning
* ordinal classification studies
* feature engineering experimentation
* imbalance learning research
* Kaggle-style ML competitions

---

# License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and adapt this work with proper attribution.

---

# MIT License

Copyright (c) 2026 Dimas Pasha Akrilian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

# Author

**Dimas Pasha Akrilian**

This repository is part of my **advanced machine learning and deep learning portfolio**, focusing on:

* tabular deep learning
* feature interaction modeling
* residual neural architectures
* ordinal classification
* imbalance-aware learning
* competition-grade machine learning pipelines
