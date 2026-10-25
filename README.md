# Telecom Customer Churn Prediction & SHAP Audit

Does fixing class imbalance with SMOTE change *why* a churn model predicts what it predicts?

This project trains Random Forest and XGBoost on the IBM Telco Customer Churn dataset, with and without SMOTE, and compares both their performance and their SHAP explanations.


## Overview

Customer churn is costly for telecom operators: keeping a customer is cheaper than acquiring a new one. Only about 26.5% of customers churn, so models tend to favor the majority class and miss many churners. SMOTE is a common fix, and SHAP is a common way to explain model predictions. In the 15 recent studies (2024-2026) we reviewed, these two topics are almost always treated separately.

**Research question:** after SMOTE improves recall, do the model's explanations stay the same?

## Dataset

- **Source:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (73.46% stay, 26.54% churn)

## Pipeline

1. **Cleaning:** converted `TotalCharges` from text to numeric, fixed 11 hidden missing values (new customers with tenure 0), dropped `customerID`, encoded the target.
2. **Encoding:** one-hot encoding of the 14 categorical columns (30 features in total).
3. **Split:** 80/20 stratified train/test split (5,634 train / 1,409 test), done before any resampling.
4. **Scaling:** Min-Max scaling of `tenure`, `MonthlyCharges` and `TotalCharges`, fitted on the training set only.
5. **SMOTE:** applied to the training set only, adding 2,644 synthetic churners (5,634 to 8,278 rows, 50/50).
6. **Models:** Random Forest (100 trees, `random_state=42`) and XGBoost, each trained on the original and on the SMOTE training set. The test set is identical for all four models.
7. **Evaluation:** precision, recall and F1 on the churn class, plus accuracy.
8. **Explainability:** SHAP `TreeExplainer`, mean absolute SHAP value per feature, compared before and after SMOTE.

## Results

### Performance (test set, churn class)

| Model | Recall | Precision | Accuracy |
|---|---|---|---|
| Random Forest, baseline | 0.50 | 0.62 | 79% |
| Random Forest, SMOTE | 0.64 | 0.55 | 77% |
| XGBoost, baseline | 0.52 | 0.59 | 78% |
| XGBoost, SMOTE | 0.67 | 0.54 | 76% |

SMOTE raises churn recall by about 14 to 15 points, at the cost of some precision and about 2 points of accuracy.

### SHAP shift after SMOTE (change in mean absolute SHAP value)

| Feature | Random Forest | XGBoost | Verdict |
|---|---|---|---|
| tenure | +4% | +8% | Stable |
| TotalCharges | -5% | -2% | Stable |
| MonthlyCharges | +113% | +125% | Unstable |
| OnlineSecurity_Yes | +95% | +148% | Unstable |
| TechSupport_Yes | +106% | +160% | Unstable |
| OnlineBackup_Yes | +126% | +313% | Unstable |
| InternetService_Fiber optic | -15% | -62% | Unstable |
| PaymentMethod_Electronic check | -37% | -61% | Unstable |

### Key findings

- SMOTE improves recall in both models, as expected.
- SMOTE also reorganizes the explanations: `tenure` and `TotalCharges` stay stable, while pricing and service features gain importance and `Fiber optic` and `Electronic check` lose it.
- The same direction appears in both algorithms, and XGBoost shifts more than Random Forest.
- A model can look better on recall while telling the business a different story about why customers leave. A before/after SHAP comparison is a useful check before deploying a resampled model.

## Limitations

- Single public dataset, so generalization to other operators is untested.
- Static snapshot: no temporal behavior is modeled.
- Single train/test split and a single random seed. Cross-validation and repeated runs would show how robust the SHAP shifts are.
- The shift is observed but its cause is not proven. Our hypothesis is that part of it comes from the synthetic samples.

## Future work

- Repeat the pipeline on other churn datasets.
- Try SMOTE variants for categorical data (SMOTE-NC, SMOTE-ENN) and class weights.
- Add cross-validation and multiple seeds.
- Include temporal features and unstructured data (complaints, call transcripts).

## Tech stack

Python, Jupyter Notebook, pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP, Matplotlib.

## Getting started

```bash
git clone https://github.com/zahrae12/<your-repo-name>.git
cd <your-repo-name>
pip install pandas scikit-learn imbalanced-learn xgboost shap matplotlib jupyter
jupyter notebook
```

Download the dataset from Kaggle (link above), place the CSV where the notebook expects it, then run the notebook cells in order.

## Authors

- Doha Chaouki
- Fatima Ezzahrae Zammou
- Hiba Chenaoui

Supervised by Pr. Khaoula Enaimi.

## License

Add a license of your choice (for example MIT), or remove this section.
