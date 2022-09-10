import pandas as pd
import numpy as np
import joblib


path = 'https://raw.githubusercontent.com/absiddik7/Datasets/main/heart_2020_cleaned.csv'
df = pd.read_csv(path)
df_c = df.copy() # keep a copy of the df

df.head()

df.info()


##Data Preprocessing

###Encoding Categorical Values
from sklearn.preprocessing import OrdinalEncoder
categorical_col = df.select_dtypes(include=['object']).columns
ordEncoder = OrdinalEncoder()
df[categorical_col] = ordEncoder.fit_transform(df[categorical_col])

df.head()

df.describe()



"""###Dataset Balancing"""

disease = df.groupby('HeartDisease').size()
disease


X = df.drop('HeartDisease',axis=1)
y = df['HeartDisease']

y.value_counts()

from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X,y)

y_resampled.value_counts()

"""#Machine Learning """

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled, test_size=0.3, random_state=42)

"""###Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print (classification_report(y_test,predictions))

joblib.dump(dtree,"heart_disease_pred_model.pkl")


"""###Random Forest Classifier"""

"""from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)
print (classification_report(y_test,rfc_prediction))"""



"""Save the Random Forest Classifier model"""
#filename = 'heart_disease_pred_model'
#pickle.dump(rfc,open(filename,'wb'))

"""Save the Random Forest Classifier model using joblib"""
#joblib.dump(rfc,"heart_disease_pred_model.pkl")