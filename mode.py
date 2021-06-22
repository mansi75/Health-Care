import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("heart.csv")
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
#a = pd.get_dummies(df['chest_pain_type'], prefix = "chest_pain_type")
#b = pd.get_dummies(df['thalassemia'], prefix = "thalassemia")
#c = pd.get_dummies(df['st_slope'], prefix = "st_slope")
#frames = [df, a, b, c]
#df = pd.concat(frames, axis = 1)
df = df.drop(columns = ['num_major_vessels', 'fasting_blood_sugar'])

y = df.target.values
np.random.shuffle(df.values)
df= df.drop(['target'], axis = 1)
sc= StandardScaler()
df=sc.fit_transform(df)

#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#x_train = x_train.T
#y_train = y_train.T
#x_test = x_test.T
#y_test = y_test.T

classifier = RandomForestClassifier(n_estimators = 1000, random_state = 1)
classifier.fit(df,y)
#pred = classifier.predict(x_test.T)
#acc = accuracy_score(pred, y_test.T)


pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()