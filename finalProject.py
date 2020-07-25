import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:/Users/TANISHQ/Desktop/ML PROJECT/train.csv')
X=dataset.iloc[:,1:].values
y=dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=200,criterion='entropy', random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
print("Accuracy of Random_Forest:",acc*100)

d=X_test[2]
d.shape=(28,28)
plt.imshow(255-d, cmap='gray')
print(clf.predict([X_test[2]]))
plt.show()

