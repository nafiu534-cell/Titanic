

# 🚢 Titanic Survival Prediction: A Machine Learning Approach

This project applies a supervised machine learning model to predict passenger survival aboard the RMS Titanic. By analyzing historical data, we use a logistic regression model to answer the question: **"Who was more likely to survive?"**

## 📊 Project Overview

The RMS Titanic disaster presents one of the most iconic datasets in data science. This project involves:

- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering and encoding  
- Model training using logistic regression  
- Performance evaluation  
- Prediction on unseen test data

## 🔍 Dataset

We used the classic Titanic dataset provided by [Kaggle](https://www.kaggle.com/competitions/titanic/data), which includes:

- Demographics: `Age`, `Sex`
- Socioeconomic info: `Pclass`, `Fare`, `Embarked`
- Family connections: `SibSp`, `Parch`
- Other: `Ticket`, `Cabin`, `Name`, `Survived` (label)

## 🛠️ Key Technologies

- Python 🐍
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn (Logistic Regression)

## 📁 Project Structure

```bash
.
├── README.md               # Project documentation
├── train.csv               # Training dataset
├── test.csv                # Test dataset (raw)
├── Test_New1.csv           # Processed test dataset with actual names
├── titanic_prediction.ipynb# Jupyter Notebook with full analysis
└── images/                 # (Optional) Visualizations for documentation
```

## ⚙️ Model Pipeline

1. **Data Cleaning**  
   - Fill missing `Age` values by median age per class  
   - Drop `Cabin` due to high missing ratio  
   - Fill `Embarked` with mode

2. **Feature Engineering**  
   - One-hot encode categorical features (`Sex`, `Pclass`, `Embarked`)  
   - Drop `Name` and `Ticket`

3. **Model Training**  
   - Logistic Regression  
   - Train-test split: 80/20  
   - Feature scaling for `Age` and `Fare`

4. **Evaluation**  
   - Accuracy: **~80.45%**  
   - Confusion matrix analysis  
   - Feature importance visualization

5. **Testing on New Data**  
   - Apply same preprocessing steps  
   - Predict and visualize survival distribution

## 📈 Key Insights

- **Gender was the strongest survival predictor** — women had significantly higher survival rates  
- **First-class passengers** were more likely to survive than those in lower classes  
- Logistic regression, while simple, offers solid interpretability and decent accuracy  

## 📷 Visuals

The project includes informative visualizations such as:

- Survival distribution by gender and class  
- Feature importance based on logistic regression coefficients  
- Survival pie charts for predicted results on test data

## 🧠 Future Work

- Try ensemble models (Random Forest, XGBoost) for improved accuracy  
- Feature extraction from `Name` (e.g., titles)  
- Use cross-validation for robust evaluation  
