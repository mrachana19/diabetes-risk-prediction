# **Predicting Diabetes Risk in PCOS/PCOD Patients**

## **Project Overview**

This project aims to address the heightened risk of diabetes in women with Polycystic Ovary Syndrome (PCOS) or Polycystic Ovarian Disease (PCOD). PCOS/PCOD is a common condition that often leads to insulin resistance, obesity, and hormonal imbalances, significantly increasing the likelihood of developing type 2 diabetes. By building a predictive model, this project provides early identification of high-risk individuals, allowing for targeted interventions to prevent diabetes onset.

## **Objectives**

- **Primary Goal**: Develop a predictive model to assess the risk of diabetes in PCOS/PCOD patients using health indicators.
- **Secondary Goals**:
  - Identify key features and interactions contributing to diabetes risk.
  - Combine different machine learning models to enhance prediction accuracy.
  - Use SHAP (SHapley Additive exPlanations) to interpret model decisions and provide actionable insights.

## **Key Features and Techniques Used**

1. **Data Preparation**:
   - Feature Engineering: Added interaction and polynomial features.
   - Standard Scaling and Train-Test Split.

2. **Model Development**:
   - Neural Network: For capturing complex patterns.
   - Random Forest and XGBoost: For handling non-linear relationships.
   - Stacking Ensemble: Combining models using Logistic Regression as a meta-model.

3. **Model Evaluation**:
   - Metrics: Accuracy, ROC-AUC, precision, recall, and f1-score.
   - SHAP: Used to identify key features influencing predictions.

## **Tools and Technologies**

- Python
- PyTorch
- Scikit-learn
- XGBoost
- SHAP
- Pandas, NumPy
- Matplotlib, Seaborn
- Logistic Regression, Random Forest
- StandardScaler, PolynomialFeatures
- Grid Search
- Jupyter Notebook

## **Results and Insights**

- **Performance**: Stacking Ensemble achieved 63% accuracy and an ROC-AUC score of 0.56.
- **Key Findings**:
  - High BMI combined with PCOS status is a major predictor of diabetes risk.
  - Insulin levels and low physical activity are critical factors.
  - Age and insulin resistance further amplify the risk.

## **Conclusion**

This project tackles a significant healthcare challenge by predicting diabetes risk in PCOS patients. The approach combines advanced machine learning techniques with clinical insights, offering valuable tools for early intervention and personalized patient care.

## **Future Work**

- Refine features based on SHAP insights.
- Experiment with more advanced ensemble methods.
- Expand the dataset for greater robustness.

## **Contributing**

Contributions are welcome! Fork this repo and submit a pull request for improvements.

