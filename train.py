"""
Train and save all ML models for Online Shoppers Purchasing Intention prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

print("Loading and preprocessing dataset...")

# Load Dataset
data = pd.read_csv("dataset.csv")

# Encode Categorical Features using separate encoders and save them
le_month = LabelEncoder()
le_visitor = LabelEncoder()
le_weekend = LabelEncoder()
le_revenue = LabelEncoder()

data["Month"] = le_month.fit_transform(data["Month"].astype(str))
data["VisitorType"] = le_visitor.fit_transform(data["VisitorType"].astype(str))
data["Weekend"] = le_weekend.fit_transform(data["Weekend"].astype(str))
data["Revenue"] = le_revenue.fit_transform(data["Revenue"].astype(str))

# Save encoders
with open('model/encoder_month.pkl', 'wb') as f:
    pickle.dump(le_month, f)
with open('model/encoder_visitor.pkl', 'wb') as f:
    pickle.dump(le_visitor, f)
with open('model/encoder_weekend.pkl', 'wb') as f:
    pickle.dump(le_weekend, f)
with open('model/encoder_revenue.pkl', 'wb') as f:
    pickle.dump(le_revenue, f)

# Split Features and Target Variable
X = data.drop("Revenue", axis=1)
y = data["Revenue"]

# Split Data into Training and Testing Sets (stratified to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use in predictions
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Training and saving models...\n")

# 1. Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
with open('model/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("✓ Logistic Regression saved\n")

# 2. Decision Tree Classifier
print("Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt.fit(X_train_scaled, y_train)
with open('model/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt, f)
print("✓ Decision Tree saved\n")

# 3. K-Nearest Neighbors
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
with open('model/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("✓ KNN saved\n")

# 4. Naive Bayes
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
with open('model/naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb, f)
print("✓ Naive Bayes saved\n")

# 5. Random Forest Classifier
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
with open('model/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("✓ Random Forest saved\n")

# 6. XGBoost Classifier
print("Training XGBoost...")
sample_weights = compute_sample_weight('balanced', y_train)
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
)
xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
with open('model/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb, f)
print("✓ XGBoost saved\n")

print("=" * 50)
print("All models trained and saved successfully!")
print("Saved models:")
print("  - model/scaler.pkl")
print("  - model/logistic_regression.pkl")
print("  - model/decision_tree.pkl")
print("  - model/knn.pkl")
print("  - model/naive_bayes.pkl")
print("  - model/random_forest.pkl")
print("  - model/xgboost.pkl")
print("=" * 50)
