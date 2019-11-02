import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

filepath = "C:/Users/Christina Yan/Downloads/cancer_collapsed_cleaned.csv"
df = pd.read_csv(filepath)
df.drop(df.columns[0:3], axis = 1, inplace = True)
df.columns

# re-index
cols = pd.read_csv("C:/Users/Christina Yan/Downloads/columns.csv", header = None)
df.columns = cols.iloc[:30,1]
df.columns


# actual stuff
X = df.iloc[1:200000,8:29]
y = df['diagnosis_region'][1:200000]



model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()
print feat_importances.nlargest(10)

# test/train set split 
train, test = train_test_split(df, test_size=0.2)


