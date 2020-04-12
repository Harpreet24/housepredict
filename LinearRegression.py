
#==============================================================================
# Import libraries
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#==============================================================================
# imort the dataset of flat prices
#==============================================================================

flatdata = pd.read_csv ('Price.csv')
X = flatdata.iloc [:,0].values
y = flatdata.iloc [:,1].values


#==============================================================================
# split the dataset into training and test set. We will use 80/20 approach
#==============================================================================
X=X.reshape(-1,1)
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = .20, 
                                                     random_state = 0)

#==============================================================================
# Fitting the Linear Regression algo to the Training set
#==============================================================================

from sklearn.linear_model import LinearRegression
regressoragent = LinearRegression()
regressoragent.fit (X_train, y_train )            
                   
#==============================================================================
# Now check what our model learned by predicting the X_test values
#==============================================================================

#predictValues = regressoragent.predict(X_test)


# Saving model to disk
pickle.dump(regressoragent, open('pricemodel.pkl','wb'))

