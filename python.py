import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


data = pd.read_csv('transaction_dataset.csv')

X = data.drop(['FLAG', 'index','address' ], axis=1)  
y = data['FLAG']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

imputer = SimpleImputer( strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

param_grid = {
'C' : [0.1 , 1 , 10],
'kernel' : ['linear', 'rbf' , 'poly'],
'gamma' : ['scale' , 'auto']
}

svc = SVC()
grid_search = GridSearchCV( svc , param_grid , cv=5 , scoring='accuracy')
grid_search.fit(X_train , y_train)

finalmodel = grid_search.best_estimator_

y_pred = finalmodel.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))