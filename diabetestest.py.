import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


dataset=pd.read_csv("diabetes.csv")
print(dataset.head())

dataset.isnull().sum()

dataset.hist(bins=10,figsize=(10,10))
plt.show()

for i in dataset.columns:
    print(i,len(dataset[dataset[i]==0]))

zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.NaN)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NaN,mean)

sns.countplot(data=dataset,x='Outcome')

sns.heatmap(dataset.corr(),annot=True)

sns.pairplot(dataset)

plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=dataset)

print(dataset['Glucose'])

X=dataset.iloc[:,0:8];
y=dataset.iloc[:,8];
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2);

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

test_scores = []
train_scores = []

for i in range(1, 27):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score * 100, list(map(lambda x: x + 1, test_scores_ind))))

sns.lineplot(x=range(1, 27), y=train_scores, marker='*', label='Train Score')
sns.lineplot(x=range(1, 27), y=test_scores, marker='o', label='Test Score')

knn = KNeighborsClassifier(11)

knn.fit(X_train, y_train)
knn.score(X_test, y_test)

y_pred=knn.predict(X_test)
y_pred

cm=confusion_matrix(y_test,y_pred)
print(cm);

print(f1_score(y_test,y_pred));

accuracy_score(y_test, y_pred)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

Y_pred=logmodel.predict(X_test)

print(f1_score(y_test,Y_pred));

accuracy_score(y_test, Y_pred)



from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Upsample the minority class (if necessary) to address class imbalance
X_upsampled, y_upsampled = resample(X_train[y_train == 1], y_train[y_train == 1],
                                    replace=True, n_samples=X_train[y_train == 0].shape[0])
X_train_balanced = np.vstack((X_train[y_train == 0], X_upsampled))
y_train_balanced = np.hstack((y_train[y_train == 0], y_upsampled))

# Define grid search for SVM model
model = SVC(class_weight='balanced')  # Using class_weight='balanced' to handle class imbalance
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
grid = dict(kernel=kernel, C=C, gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1', error_score=0)
grid_result = grid_search.fit(X_train_balanced, y_train_balanced)
svm_pred = grid_result.predict(X_test)

print("SVM Classifier:")
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))

# Print the best hyperparameters found by the grid search
print("Best hyperparameters:")
print(grid_search.best_params_)

# Get the confusion matrix
cm = confusion_matrix(y_test, svm_pred)
print("Confusion Matrix:")
print(cm)


# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)

print("Random Forest Classifier:")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

print("Decision Tree Classifier:")
print(classification_report(y_test, dt_pred))
print("Accuracy:", accuracy_score(y_test, dt_pred))








