import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  LabelEncoder , OneHotEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


data = pd.read_csv('transaction_dataset.csv')

X = data.drop(['FLAG', 'Index','Address' ], axis=1)  
y = data['FLAG']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numerical_cols = X_train.select_dtypes(include =['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
   transformers= [
       ('nums' , SimpleImputer(strategy='mean'), numerical_cols ),
       ('cats', OneHotEncoder(handle_unknown='ignore'), categorical_cols )
   ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

param_grid = {
'C' : [1],
'kernel' : ['rbf'],
'gamma' : ['scale']
}

svc = SVC()
grid_search = GridSearchCV( svc , param_grid , cv=5 , scoring='accuracy')
grid_search.fit(X_train , y_train)

finalmodel = grid_search.best_estimator_

y_pred = finalmodel.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


