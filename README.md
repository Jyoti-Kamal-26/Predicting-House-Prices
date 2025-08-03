# California Housing Price Prediction

Hi! This is a machine learning project I worked on using the California Housing dataset. The main goal was to predict house prices based on features like average rooms, population, and location. I explored the data, trained a few models, and evaluated their performance using different metrics. It was a great hands-on experience in applying regression techniques and data preprocessing.

---

## About the Project

* **Dataset**: California Housing (from Scikit-learn)
* **Objective**: Predict median house values using numerical and location-based features.
* **Tools & Libraries**: Python, Pandas, Seaborn, Matplotlib, Scikit-learn

---

## What I Did

1. **Loaded and Prepared the Data**
   Used Scikit-learn’s built-in California housing dataset and converted it into a Pandas DataFrame. Added the target variable (`Price`) and checked for missing values.

2. **Exploratory Data Analysis (EDA)**
   Explored the relationships between variables, looked at correlations, and visualized outliers using boxplots.

3. **Data Splitting & Scaling**
   Split the data into training and testing sets. Used `StandardScaler` to normalize the feature values.

4. **Model Building**
   Trained a basic Linear Regression model. Also added Lasso and Ridge Regression to compare performance and reduce overfitting.

5. **Model Evaluation**
   Evaluated using MAE, MSE, RMSE, and R² Score. Also plotted residuals to understand prediction errors better.

6. **Model Saving**
   Saved the final model using Pickle so it can be reused without retraining.

---
