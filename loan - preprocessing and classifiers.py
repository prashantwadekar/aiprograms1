
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#######data cleaning###########
df = pd.read_csv('E:\ARTI_INT\Code\loan.csv')

#drop unnecesary colums
df = df.drop(columns = ['name'])

#one hot encode required columns
onehot_age = pd.get_dummies(df['age'])
df = df.drop(columns = ['age']).join(onehot_age)

onehot_cr = pd.get_dummies(df['credit_rating'])
df = df.drop(columns = ['credit_rating']).join(onehot_cr)

#binarize to 0 and 1
df['has_job'], mapping_has_job = df['has_job'].factorize()
df['own_house'], mapping_own_house = df['own_house'].factorize()
df['class'], mapping_class = df['class'].factorize()

#resacle to normalize
scaler = preprocessing.MinMaxScaler()
df['salary'] = scaler.fit_transform(df[['salary']])

#train test split
y = df['class'].values
X = df.drop(columns=['class']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#######prediction###########

#Decision Tree
print('Decision Trees')
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('pred: ', clf.predict(X_test))
print('actual: ', y_test)

#SVM
print('SVM')
clf = SVC()
clf.fit(X_train, y_train)
print('pred: ', clf.predict(X_test))
print('actual: ', y_test)

#Random Forest
print('Random Forests')
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('pred: ', clf.predict(X_test))
print('actual: ', y_test)
