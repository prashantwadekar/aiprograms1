import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00466/thoracic_surgery.csv"
data = pd.read_csv(url)

# a. Calculate basic statistical summary for numeric variables for survived and not survived groups separately
survived_summary = data[data['Risk1Yr'] == 0].describe()
not_survived_summary = data[data['Risk1Yr'] == 1].describe()

print("a. Statistical Summary for Survived Patients:")
print(survived_summary)

print("\na. Statistical Summary for Not Survived Patients:")
print(not_survived_summary)

# b. Calculate the mean of Age for patients with Type 2 DM
mean_age_dm = data[data['PRE17'] == 1]['Age'].mean()
print(f"\nb. Mean Age for patients with Type 2 DM (PRE17): {mean_age_dm:.2f} years")



# c. Analyze various features
# Additional analysis based on specific columns

# Visualize Forced Vital Capacity (PRE4)
plt.figure(figsize=(10, 6))
sns.histplot(data['PRE4'], kde=True)
plt.title('Distribution of Forced Vital Capacity (PRE4)')
plt.xlabel('Forced Vital Capacity (PRE4)')
plt.ylabel('Frequency')
plt.show()

# Visualize Volume that has been exhaled (PRE5)
plt.figure(figsize=(10, 6))
sns.histplot(data['PRE5'], kde=True)
plt.title('Distribution of Volume that has been exhaled (PRE5)')
plt.xlabel('Volume that has been exhaled (PRE5)')
plt.ylabel('Frequency')
plt.show()

# Visualize Performance Status (PRE6)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE6'])
plt.title('Frequency Distribution of Performance Status (PRE6)')
plt.xlabel('Performance Status (PRE6)')
plt.ylabel('Frequency')
plt.show()

# Visualize Pain before surgery (PRE7)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE7'])
plt.title('Frequency Distribution of Pain before surgery (PRE7)')
plt.xlabel('Pain before surgery (PRE7)')
plt.ylabel('Frequency')
plt.show()

# Visualize Dyspnoea before surgery (PRE8)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE8'])
plt.title('Frequency Distribution of Dyspnoea before surgery (PRE8)')
plt.xlabel('Dyspnoea before surgery (PRE8)')
plt.ylabel('Frequency')
plt.show()

# Visualize Cough before surgery (PRE9)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE9'])
plt.title('Frequency Distribution of Cough before surgery (PRE9)')
plt.xlabel('Cough before surgery (PRE9)')
plt.ylabel('Frequency')
plt.show()

# Visualize Weakness before surgery (PRE10)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE10'])
plt.title('Frequency Distribution of Weakness before surgery (PRE10)')
plt.xlabel('Weakness before surgery (PRE10)')
plt.ylabel('Frequency')
plt.show()

# Visualize Size of the original tumour (PRE14)
plt.figure(figsize=(10, 6))
sns.histplot(data['PRE14'], kde=True)
plt.title('Distribution of Size of the original tumour (PRE14)')
plt.xlabel('Size of the original tumour (PRE14)')
plt.ylabel('Frequency')
plt.show()

# Visualize Diabetes Mellitus (PRE17)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE17'])
plt.title('Frequency Distribution of Diabetes Mellitus (PRE17)')
plt.xlabel('Diabetes Mellitus (PRE17)')
plt.ylabel('Frequency')
plt.show()

# Visualize Peripheral Arterial Diseases (PRE19)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE19'])
plt.title('Frequency Distribution of Peripheral Arterial Diseases (PRE19)')
plt.xlabel('Peripheral Arterial Diseases (PRE19)')
plt.ylabel('Frequency')
plt.show()

# Visualize Smoking (PRE25)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE25'])
plt.title('Frequency Distribution of Smoking (PRE25)')
plt.xlabel('Smoking (PRE25)')
plt.ylabel('Frequency')
plt.show()

# Visualize Asthma (PRE30)
plt.figure(figsize=(8, 6))
sns.countplot(data['PRE30'])
plt.title('Frequency Distribution of Asthma (PRE30)')
plt.xlabel('Asthma (PRE30)')
plt.ylabel('Frequency')
plt.show()

# Visualize Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# ... (remaining code)

# c. Analyze various features
# You can print specific columns or perform more detailed analysis based on your requirements

# d. Frequency plot for "Risk1Yr"
plt.figure(figsize=(8, 6))
sns.countplot(data['Risk1Yr'])
plt.title('Frequency Plot for "Risk1Yr"')
plt.xlabel('Risk1Yr')
plt.ylabel('Frequency')
plt.show()

# e. Frequency distribution of DGN
plt.figure(figsize=(12, 6))
sns.countplot(data['DGN'])
plt.title('Frequency Distribution of DGN')
plt.xlabel('DGN')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
