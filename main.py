import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  LabelEncoder , OneHotEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

print("Starting program...")
print("Loading dataset...")
data = pd.read_csv('transaction_dataset.csv')
print(f"Dataset loaded. Shape: {data.shape}")

print("Preparing features...")
X = data.drop(['FLAG', 'Index','Address' ], axis=1)  
y = data['FLAG']  
print("Features prepared.")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

print("Identifying column types...")
numerical_cols = X_train.select_dtypes(include =['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns")

print("Setting up preprocessor...")
preprocessor = ColumnTransformer(
   transformers= [
       ('nums' , SimpleImputer(strategy='mean'), numerical_cols ),
       ('cats', OneHotEncoder(handle_unknown='ignore'), categorical_cols )
   ]
)

print("Preprocessing training data...")
X_train = preprocessor.fit_transform(X_train)
print("Preprocessing test data...")
X_test = preprocessor.transform(X_test)
print("Preprocessing complete.")

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("SMOTE complete.")

print("Encoding labels...")
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
print("Label encoding complete.")

param_grid = {
'C' : [1],
'kernel' : ['rbf'],
'gamma' : ['scale']
}

print("Starting grid search...")
svc = SVC()
grid_search = GridSearchCV( svc , param_grid , cv=5 , scoring='accuracy')
grid_search.fit(X_train , y_train)
print("Grid search complete.")

finalmodel = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

print("Making predictions...")
y_pred = finalmodel.predict(X_test)

print("\n=== Final Results ===")
print("✅ Data Preprocessing Completed")
print("✅ SMOTE Resampling Completed")
print("✅ Model Training Completed")
print("✅ Prediction Completed")
print("Accuracy:", accuracy_score(y_test, y_pred))


