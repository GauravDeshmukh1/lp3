import numpy as np
import pandas as pd

#reading dataset
dataset = pd.read_csv("decision_Tree_dataset.csv")

#perform label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
dataset = dataset.apply(LabelEncoder().fit_transform)
print(dataset)

x = dataset.iloc[ : ,  : -1]
y = dataset['Buys']

x = np.asarray(x)

# applying decision tree classifier
from sklearn.tree import DecisionTreeClassifier as DTC

c = DTC()
c = c.fit(x, y)

#predicting
c.predict([[2, 0, 1, 0],[1,0,0,0]])
