import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Crop_Production (2).csv')

dataset['Crop_Production'] = dataset['Crop_Production'].replace(["low", "high"], [0,1])

from sklearn.preprocessing import LabelEncoder
st = LabelEncoder()
ct = LabelEncoder()
fn = LabelEncoder()
r = LabelEncoder()
cp = LabelEncoder()


dataset['Soil_Type'] = st.fit_transform(dataset['Soil_Type'])
dataset['Crop_Type'] = ct.fit_transform(dataset['Crop_Type'])
dataset['Fertilizer_Name'] = fn.fit_transform(dataset['Fertilizer_Name'])  
dataset['Rainfall'] = r.fit_transform(dataset['Rainfall'])
#dataset['Crop_Production'] = cp.fit_transform(dataset['Crop_Production'])


x = dataset.iloc[:,0:11].values
y = dataset.iloc[:,11].values


x[:,3].max()
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [3])
x = one.fit_transform(x).toarray()
x= x[:,1:]

x[:,8].max()
from sklearn.preprocessing import OneHotEncoder
one1 = OneHotEncoder(categorical_features = [8])
x = one1.fit_transform(x).toarray()
x= x[:,1:]

x[:,26].max()
from sklearn.preprocessing import OneHotEncoder
one2 = OneHotEncoder(categorical_features = [26])
x = one2.fit_transform(x).toarray()
x= x[:,1:]

x[:,31].max()
from sklearn.preprocessing import OneHotEncoder
one3 = OneHotEncoder(categorical_features = [31])
x = one3.fit_transform(x).toarray()
x= x[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling



# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred1 = classifier.predict([[0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,26,52,38,37,0,0,3]])
y_pred2=classifier.predict([[0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,36,60,43,15,43,41,9]])
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

import seaborn as sns
g=sns.PairGrid(dataset)
g.map(plt.scatter)

plt.figure(figsize=(10,5))
sns.heatmap(dataset.corr(),annot=True)


plt.boxplot(dataset["Crop_Production"])
sns.pairplot(dataset)
