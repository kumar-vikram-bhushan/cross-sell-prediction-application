import pandas as pd
import seaborn as sns
import numpy as np

data=pd.read_csv("health-insurance.csv")
data.head()
data.drop(['id'],axis=1,inplace=True)
# data.drop(['Vintage'],axis=1,inplace=True)

data['Gender']=data['Gender'].replace({"Male" : 0,"Female":1})
data['Vehicle_Age']=data['Vehicle_Age'].replace("1-2 Year",3)
data['Vehicle_Age']=data['Vehicle_Age'].replace("< 1 Year",1)
data['Vehicle_Age']=data['Vehicle_Age'].replace("> 2 Years",2)

data['Vehicle_Damage']=data['Vehicle_Damage'].replace({"Yes":1,"No":0})



for i in range(len(data['Region_Code'])):
    if(i>=0 and i<10):
        data['Region_Code']=data['Region_Code'].replace({i:0})
    elif(i>=10 and i<20):
        data['Region_Code']=data['Region_Code'].replace({i:1})
    elif(i>=20 and i<30):
        data['Region_Code']=data['Region_Code'].replace({i:2})
    elif(i>=30 and i<40):
        data['Region_Code']=data['Region_Code'].replace({i:3})
    elif(i>=40 and i<=50):
        data['Region_Code']=data['Region_Code'].replace({i:4})

for i in range(len(data['Policy_Sales_Channel'])):
    if(i>=0 and i<30):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:0})
    elif(i>=30 and i<60):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:1})
    elif(i>=60 and i<120):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:2})
    elif(i>=120 and i<150):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:3})
    elif(i>=150 and i<=160):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:4})
Y = data['Response']
X= data.drop('Response',axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier(criterion='gini',max_features=10,max_depth=5)
DTC=DTC.fit(X_train,y_train)
pred=DTC.predict(X_test)
prob=DTC.predict_proba(X_test)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr3, tpr3, thresh3 = roc_curve(y_test, prob[:,1], pos_label=1)
auc_score3= roc_auc_score(y_test, prob[:,1])

print(auc_score3)
import pickle
pickle.dump(DTC, open('model.pkl','wb'))
loaded_model = pickle.load(open('model.pkl','rb'))
result = loaded_model.score(X_test,y_test)
print(X_train.head())