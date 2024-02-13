


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


glass = pd.read_csv("glass.csv")


X = glass.iloc[:, :-1]
y = glass.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


gnb = GaussianNB()
gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score


glass = pd.read_csv("glass.csv")


X = glass.iloc[:, :-1]
y = glass.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, y_train)


y_pred = svm_linear.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))








# In[ ]:



