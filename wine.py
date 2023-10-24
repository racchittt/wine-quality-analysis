import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle

warnings.filterwarnings("ignore")
wine_dataset = pd.read_csv('winequality.csv')

#Preprocessing 
X= wine_dataset.drop('quality', axis=1)

#label Binarization
Y= wine_dataset['quality'] #.apply(lambda y_value: 1 if y_value>= 7 else 0)

#test train split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=3)

#Model selection and training
model= RandomForestClassifier()
model.fit(X_train, Y_train)

#Predictive system
input_data= (7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0)

input_data_as_numpy_array= np.asarray(input_data)

#reshaping the data as we are predicting the label for only one instance
input_data_reshaped= input_data_as_numpy_array.reshape( 1,-1 )

prediction= model.predict( input_data_reshaped)

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


