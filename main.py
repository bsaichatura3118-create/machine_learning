# Online Shoppers Purchasing Intentiontention

#Import Required Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import warnings
warnings.filterwarnings("ignore")   


# Load Dataset
data = pd.read_csv("dataset.csv")

#Encode Categorical Features
label_encoder = LabelEncoder()

data["Month"] = label_encoder.fit_transform(data["Month"])
data["VisitorType"] = label_encoder.fit_transform(data["VisitorType"])
data["Weekend"] = label_encoder.fit_transform(data["Weekend"])
data["Revenue"] = label_encoder.fit_transform(data["Revenue"])

# Split Features and Target Variable
X = data.drop("Revenue", axis=1)
y = data["Revenue"]
# Split Data into Training and Testing Sets (stratified to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
lr_mcc = matthews_corrcoef(y_test, lr.predict(X_test))
print("Logistic Regression MCC:", lr_mcc)

print("\n")

# 2. Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
dt_mcc = matthews_corrcoef(y_test, dt.predict(X_test))
print("Decision Tree MCC:", dt_mcc)

print("\n")

# 3. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
knn_mcc = matthews_corrcoef(y_test, knn.predict(X_test))
print("KNN MCC:", knn_mcc)


print("\n")

# 4. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]

print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
nb_mcc = matthews_corrcoef(y_test, nb.predict(X_test))
print("Naive Bayes MCC:", nb_mcc)

print("\n")

# 5. Random Forest Classifier


rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
rf_mcc = matthews_corrcoef(y_test, rf.predict(X_test))
print("Random Forest MCC:", rf_mcc)


print("\n")

# 6. XGBoost Classifier
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
)

xgb.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

print("\nXGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
xgb_mcc = matthews_corrcoef(y_test, xgb.predict(X_test))
print("XGBoost MCC:", xgb_mcc)
print("\n")
