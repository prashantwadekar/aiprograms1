import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/priya-dwivedi/IT-Projects/main/Employee_attrition.csv"
data = pd.read_csv(url)

# a. Features Analysis
print("a. Features Analysis:")
print(data.head())

# b. Basic Statistical Summary
print("\nb. Basic Statistical Summary:")
print(data.describe())

# c. Pivot table and pivot chart
pivot_table = data.pivot_table(index='Department', columns='Attrition', values='Age', aggfunc='count')
pivot_table.plot(kind='bar', stacked=True)
plt.title('Pivot Table and Pivot Chart')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

# d. Hypothesis testing (Example: t-test for 'MonthlyIncome' between Attrition groups)
from scipy.stats import ttest_ind

attrition_yes = data[data['Attrition'] == 'Yes']['MonthlyIncome']
attrition_no = data[data['Attrition'] == 'No']['MonthlyIncome']

t_statistic, p_value = ttest_ind(attrition_yes, attrition_no)
print(f"\nd. Hypothesis Testing (t-test for MonthlyIncome):")
print(f"   t-statistic: {t_statistic:.2f}, p-value: {p_value:.4f}")

# e. Variable importance using Random Forest
X = data[['Age', 'Gender', 'MaritalStatus', 'Education', 'Department', 'JobRole', 'MonthlyIncome', 'JobSatisfaction']]
y = data['Attrition']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Train RandomForestClassifier to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = rf.feature_importances_
feature_names = X.columns

# Sort features by importance
sorted_idx = feature_importance.argsort()[::-1]
feature_names_sorted = feature_names[sorted_idx]
feature_importance_sorted = feature_importance[sorted_idx]

print("\ne. Variable Importance using Random Forest:")
for feature, importance in zip(feature_names_sorted, feature_importance_sorted):
    print(f"   {feature}: {importance:.4f}")

# f. Machine Learning model to predict employee attrition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nf. Machine Learning Model (Random Forest) Results:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Classification Report:")
print(classification_rep)
