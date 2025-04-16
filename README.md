# 💳 Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent credit card transactions using various machine learning models. The dataset is highly imbalanced, making it a great case study for anomaly detection.

## 📁 Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 fraud cases (~0.172%)
- Features are PCA-transformed due to confidentiality, except `Time` and `Amount`

## 🔧 Techniques Used

- **StandardScaler** for feature scaling
- **SMOTE** to balance the dataset
- **Train-Test Split** with stratification
- **Model Evaluation** with:
  - Confusion Matrix
  - ROC Curve
  - Precision, Recall, F1-Score
  - Accuracy

## 🤖 Models Trained

- Logistic Regression
- Random Forest
- AdaBoost
- XGBoost

## 📊 Results

| Model            | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | Accuracy |
|------------------|-------------------|----------------|------------------|----------|
| Logistic Regression | 0.06              | 0.88           | 0.12             | 0.98     |
| Random Forest       | **0.87**          | 0.79           | **0.83**         | **1.00** |
| AdaBoost            | 0.10              | **0.86**       | 0.18             | 0.99     |
| XGBoost             | 0.76              | 0.80           | 0.78             | **1.00** |

## 📌 Conclusion

Random Forest provided the best balance between precision and recall. XGBoost also performed well and could be further tuned. The project showcases various ML techniques to handle real-world data challenges.

## 📚 Learnings

- Handling imbalanced data using SMOTE
- Importance of evaluation metrics beyond accuracy
- Visualization with confusion matrices and ROC curves
- Comparing multiple models with automated metric extraction

## 👨‍💻 Author

**Abiodun Amos Olaoluwa**  
[LinkedIn](https://www.linkedin.com/in/abiodun-amos) | Data Scientist | Electrical Engineer in training
