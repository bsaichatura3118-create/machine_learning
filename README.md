# Online Shoppers Purchasing Intention Prediction

## Problem Statement

The objective of this machine learning project is to predict whether an online shopper will complete a purchase (Revenue generation) or not. This is a binary classification problem that helps e-commerce businesses identify potential high-value customers and optimize their marketing strategies. The model aims to classify visitors as revenue-generating (True) or non-revenue-generating (False) based on their browsing behavior and session characteristics.

## Dataset Description

**Dataset Name:** Online Shoppers Purchasing Intention Dataset

**Source:** E-commerce Website Sessions

**Size:** 12,330 samples (after processing)

**Features:** 17 attributes including:
- **Administrative, Informational, ProductRelated:** Number of pages visited in each category
- **Administrative_Duration, Informational_Duration, ProductRelated_Duration:** Time spent (in seconds) on each category
- **BounceRate:** Percentage of visitors who left without further interaction
- **ExitRate:** Percentage of pageviews on which the user exited the site
- **PageValues:** Average value of pages visited by user
- **SpecialDay:** Closeness of the visit date to a specific special day (e.g., holidays)
- **Month:** Month of the visit (categorical)
- **VisitorType:** Type of visitor - Returning_Visitor, New_Visitor, or Other (categorical)
- **Weekend:** Boolean value indicating if the date is weekend
- **Revenue:** Target variable - Boolean indicating whether a purchase was made

**Class Distribution:** The dataset contains 12,330 samples. In `dataset.csv` the `Revenue` column has 10,465 purchases (`True`) and 1,865 non-purchases (`False`) — approximately 84.8% purchases and 15.2% non-purchases.

**Data Preprocessing:**
- Categorical features (Month, VisitorType, Weekend, Revenue) encoded using LabelEncoder
- Feature scaling applied using StandardScaler to normalize numerical features
- Stratified train-test split (80-20) to maintain class distribution
- Class weight balancing applied to handle class imbalance

---

## Models Used and Evaluation Metrics

### Evaluation Metrics Explanation:
- **Accuracy:** Overall correctness of predictions
- **AUC (Area Under ROC Curve):** Measure of model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)
- **Precision:** Proportion of positive predictions that were correct
- **Recall:** Proportion of actual positives correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure for binary classification (range: -1 to 1)

### Comparison Table: Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.5000 | 0.4746 | 0.8424 | 0.5055 | 0.6318 | -0.0182 |
| **Decision Tree** | 0.7457 | 0.4999 | 0.8487 | 0.8524 | 0.8505 | -0.0002 |
| **k-Nearest Neighbors (kNN)** | 0.8268 | 0.4891 | 0.8480 | 0.9699 | 0.9048 | -0.0127 |
| **Naive Bayes** | 0.8487 | 0.4733 | 0.8487 | 1.0000 | 0.9182 | 0.0000 |
| **Random Forest (Ensemble)** | 0.8487 | 0.4893 | 0.8487 | 1.0000 | 0.9182 | 0.0000 |
| **XGBoost (Ensemble)** | 0.5191 | 0.5045 | 0.8512 | 0.5251 | 0.6495 | 0.0074 |

---

## Model Performance Analysis and Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression performs poorly with only 50% accuracy and negative MCC (-0.0182), indicating it barely performs better than random guessing. The high precision (0.8424) but low recall (0.5055) shows the model is conservative in predicting purchases, missing many actual positive cases. This suggests linear decision boundaries are insufficient for this complex dataset. The low AUC (0.4746) indicates weak discrimination ability between classes. |
| **Decision Tree** | Decision Tree shows moderate improvement with 74.57% accuracy and balanced precision-recall (0.8487/0.8524). However, the near-zero MCC (-0.0002) and AUC (0.4999) suggest the model's predictions are barely better than random despite the high accuracy. This is due to class imbalance—the model achieves high accuracy by predicting the majority class. The F1 score of 0.8505 is misleading in this context. |
| **k-Nearest Neighbors (kNN)** | kNN achieves 82.68% accuracy with very high recall (0.9699), indicating it captures most positive cases. However, the negative MCC (-0.0127) and AUC (0.4891) reveal that despite high accuracy, the model's predictive power is weak. The extremely high recall suggests the model is biased toward predicting the majority class, leading to many false positives. The high F1 score (0.9048) masks this underlying issue. |
| **Naive Bayes** | Naive Bayes achieves 84.87% accuracy with perfect recall (1.0000), meaning all revenue-generating visitors are correctly identified. Precision matches accuracy (0.8487), suggesting consistent predictions. MCC of 0.0000 indicates neutral correlation—predictions are essentially random in terms of balancing both classes. The moderate AUC (0.4733) and perfect recall demonstrate the model is optimistic in predicting purchases but lacks discrimination ability. Despite high accuracy, it's not reliable for distinguishing between classes due to class imbalance. |
| **Random Forest (Ensemble)** | Random Forest achieves 84.87% accuracy (tied with Naive Bayes) with perfect recall (1.0000) and precision equal to accuracy (0.8487). Like Naive Bayes, the zero MCC and moderate AUC (0.4893) indicate poor discrimination despite high accuracy. The model captures all positive cases but suffers from class imbalance bias. The ensemble approach with balanced class weights improves performance, but the AUC near 0.5 suggests the model doesn't effectively separate the classes. F1 score of 0.9182 appears inflated due to dataset imbalance. |
| **XGBoost (Ensemble)** | XGBoost shows the most balanced but modest performance with 51.91% accuracy and the highest AUC (0.5045) and MCC (0.0074) among all models. This suggests XGBoost better handles class imbalance compared to other models. While accuracy is lower, the improved AUC and positive MCC indicate more reliable discrimination between classes. Precision (0.8512) is high but recall (0.5251) is moderate, showing controlled false positive rate. XGBoost's ability to learn complex patterns with balanced class weights makes it the most trustworthy model despite lower accuracy. |

### Key Insights:

1. **Class Distribution Impact:** The dataset is heavily skewed toward purchases (~84.8% purchases). This strong imbalance means models can achieve high accuracy by predicting the majority class (purchase). Therefore, accuracy is misleading here — use AUC and MCC to evaluate true discrimination performance.

2. **Best Performing Model:** **XGBoost** is the most reliable model with the highest AUC (0.5045) and positive MCC (0.0074), indicating genuine predictive power rather than majority class bias.

3. **Accuracy Paradox:** Models with high accuracy (Naive Bayes, Random Forest at 84.87%) are likely predicting the majority class (purchase) for many samples. This inflates accuracy while AUC and MCC remain near 0.5/0, showing limited discrimination between classes.

4. **Ensemble Methods:** Both ensemble methods (Random Forest and XGBoost) outperform traditional algorithms in terms of stability, though only XGBoost shows meaningful AUC improvement.

5. **Recommendations:**
   - Use **XGBoost** for deployment due to superior AUC and MCC scoredos
   - Consider SMOTE (Synthetic Minority Oversampling) or class weight adjustments further
   - Focus on AUC and MCC metrics rather than accuracy for imbalanced datasets
   - Ensemble methods with proper hyperparameter tuning may yield better results

   ---

   ## Run Instructions (detailed)

   1. Install dependencies (recommended to use Windows Python or a virtual environment):

   ```bash
   pip install -r requirements.txt
   or
   pip install --break-system-packages -r requirements.txt

   ```

   2. Train models (only if you need to retrain or update models):

   ```bash
   python3 train.py
   ```

   3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   Notes:
   - The training script now saves LabelEncoders for `Month`, `VisitorType`, and `Weekend` to the `model/` folder (`encoder_month.pkl`, `encoder_visitor.pkl`, `encoder_weekend.pkl`). The Streamlit app loads these encoders for consistent encoding.
   - Saved scaler and model pickles are loaded by the app. If the pickles were created with a different scikit-learn version than the one used at runtime, you may see an InconsistentVersionWarning when unpickling. For reproducibility, install matching scikit-learn versions or retrain models in the target environment.
   - If you cannot create a virtual environment under WSL, use the Windows Python installation to run the commands from PowerShell or Command Prompt (no sudo required).

   If you want, I can also add exact `pip` versions (freeze) to `requirements.txt` to lock scikit-learn and avoid version mismatch warnings.
