import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('transaction_dataset.csv')

imputer = SimpleImputer( strategy='constant' , fill_value = 2)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.3 , random_state=42 )
