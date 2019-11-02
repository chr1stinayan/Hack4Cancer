import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

filepath = "C:/Users/Frederick/Documents/Hack4Cancer/cancer_collapsed_cleaned.csv"
df = pd.read_csv(filepath)
df.drop(df.columns[0:3], axis = 1, inplace = True)
df.columns

# re-index
cols = pd.read_csv("C:/Users/Frederick/Documents/Hack4Cancer/columns.csv", header = None)
df.columns = cols.iloc[:30,1]
df.columns

# actual stuff
X = df.iloc[1:50000,8:29]
y = df['cancer'][1:50000]

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()