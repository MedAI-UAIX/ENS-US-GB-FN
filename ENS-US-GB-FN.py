# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 21:08:56 2022

@author: HHT
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB  
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE

KNN = KNeighborsClassifier()
RF=RandomForestClassifier()
LR=LogisticRegression()
SVM=SVC()
GNB=GaussianNB()
Bag=BaggingClassifier()
Ada=AdaBoostClassifier()
GBDT=GradientBoostingClassifier()
SGD=SGDClassifier()
DT = DecisionTreeClassifier()

model_list=[('rf',RF),('gbdt',GBDT),('bag',Bag),('dt',DT), ('lr',LR),('gnb',GNB),('svm',SVM),('sgd',SGD),('knn',KNN),('ada',Ada)]

# In[*]               
os.chdir(r'D:\\liver biopsy\statistics')
Data = pd.read_csv('./data/develop.csv')
TEST = pd.read_csv('./data/test.csv')

def typicalsamling(group,typicalNDict):
    name=group.name
    n=typicalNDict[name]
    return group.sample(n=n)

typicalNDict={0:582,1:66} #7:3
TRAIN=Data.groupby(
        "class_1='failure'_0='success'",group_keys=False
        ).apply(typicalsamling,typicalNDict)

TRAIN_index = TRAIN.index.to_list()
VAL = Data[~Data.index.isin(TRAIN_index)]
x_train = TRAIN.iloc[:, 1:-1];y_train= TRAIN.iloc[:, -1]
x_val=VAL.iloc[:, 1:-1];y_val= VAL.iloc[:, -1]
x_test=TEST.iloc[:, 2:-1];y_test= TEST.iloc[:, -1]

# In[*] 
from sklearn.base import clone
for name,model in model_list:
    score_max=0
    for step in range(1000):
        clone_model = clone(model)          

        feature_scores = []
        for n in range(5,15):
            selector = RFE(clone_model,n_features_to_select=n,step=1).fit(x_train,y_train)
            score = cross_val_score(clone_model, selector, y_train, cv=5)
            if score > score_max:
                score_max=score
                best_n_features=n
                selector = RFE(clone_model,n_features_to_select=best_n_features,step=1).fit(x_train,y_train)
                df = pd.Series(selector.support_,index = list(x_train.columns))
                selected_features = df[df==True].index

                joblib.dump(clone_model, './{}.model'.format(name))
                print(name,selected_features)

#Comput following best 4 models and corresponding features
name1,model1,selected_features1='rf',RF,['Ca_Hepatitis_1="yes"_0="no"', 'Ca_Hepatobiliary_surgery_1="yes"_0="no"', 'Ca_Tumor_Marker', 'Ca_Puncture_depth_1=">8"_0="<=8"', 'Ca_Number_of_puncture_1=">=3"_0="<3"', "Ca_Lesion_location_leftlobe_1='yes'_0='no'", 'Ca_Cirrosis_1="yes"_0="no"', 'Ca_Ascites_1="yes"_0="no"', 'Ca_tumor_size_1="<2cm"_0=">=2"cm', "Ca_Visual_score_C_1='yes'_0='no'", 'Ca_Near_great_vessels_1="yes"_0="no"']
name2,model2,selected_features2='gbdt',GBDT,['Ca_Work_experience_1="<=3"_0=">3"', 'Ca_Near_great_vessels_1="yes"_0="no"', 'Ca_Tumor_Marker', 'Ca_Number_of_puncture_1=">=3"_0="<3"', 'Ca_Ascites_1="yes"_0="no"', 'Ca_Cirrosis_1="yes"_0="no"', "Ca_Lesion_location_leftlobe_1='yes'_0='no'", 'Ca_System_therapy_1="yes"_0="no"', 'Ca_Hepatobiliary_surgery_1="yes"_0="no"', 'Ca_Hepatic_parenchyma_coarseness_1="yes"_0="no"', 'Ca_tumor_size_1="<2cm"_0=">=2"cm', 'Ca_Hepatitis_1="yes"_0="no"']
name3,model3,selected_features3='bag',Bag,["Ca_Visual_score_C_1='yes'_0='no'", 'Ca_Hepatitis_1="yes"_0="no"', 'Ca_Cirrosis_1="yes"_0="no"', 'Ca_tumor_size_1="<2cm"_0=">=2"cm', 'Ca_Puncture_depth_1=">8"_0="<=8"', 'Ca_Near_great_vessels_1="yes"_0="no"', 'Ca_Hepatic_parenchyma_coarseness_1="yes"_0="no"', 'Ca_Number_of_puncture_1=">=3"_0="<3"', 'Ca_Work_experience_1="<=3"_0=">3"', "Ca_Lesion_location_leftlobe_1='yes'_0='no'", 'Ca_Tumor_Marker']
name4,model4,selected_features4='dt',DT,["Ca_Visual_score_C_1='yes'_0='no'", 'Ca_Near_great_vessels_1="yes"_0="no"', 'Ca_tumor_size_1="<2cm"_0=">=2"cm', 'Ca_Cirrosis_1="yes"_0="no"', 'Ca_Hepatic_parenchyma_coarseness_1="yes"_0="no"', 'Ca_Puncture_depth_1=">8"_0="<=8"', 'Ca_Number_of_puncture_1=">=3"_0="<3"', 'Ca_Tumor_Marker', 'Ca_Work_experience_1="<=3"_0=">3"', "Ca_Child_Pugh_class_C_1='yes'_0='no'", 'Ca_Hepatitis_1="yes"_0="no"']
# In[*]
model=joblib.load('./{}.model'.format('rf'));selected_features=selected_features1
if hasattr(model, "predict_proba"):
    probas_train = model.predict_proba(x_train[selected_features])
    probas_train1 = probas_train[:, 1]
elif hasattr(model, "decision_function"):
    probas_train1 = model.decision_function(x_train[selected_features])

model=joblib.load('./{}.model'.format('gbdt'));selected_features=selected_features2
if hasattr(model, "predict_proba"):
    probas_train = model.predict_proba(x_train[selected_features])
    probas_train2 = probas_train[:, 1]
elif hasattr(model, "decision_function"):
    probas_train2 = model.decision_function(x_train[selected_features])

model=joblib.load('./{}.model'.format('bag'));selected_features=selected_features3
if hasattr(model, "predict_proba"):
    probas_train = model.predict_proba(x_train[selected_features])
    probas_train3 = probas_train[:, 1]
elif hasattr(model, "decision_function"):
    probas_train3 = model.decision_function(x_train[selected_features])

model=joblib.load('./{}.model'.format('dt'));selected_features=selected_features4
if hasattr(model, "predict_proba"):
    probas_train = model.predict_proba(x_train[selected_features])
    probas_train4 = probas_train[:, 1]
elif hasattr(model, "decision_function"):
    probas_train4 = model.decision_function(x_train[selected_features])

auc_temp=0;result_temp=[]
for i in range(1,8):
    for j in range(1,10-i):
        for k in range(1,10-i-j):
           probas_train_combine = pd.Series(probas_train1)*i/10+pd.Series(probas_train2)*j/10+pd.Series(probas_train3)*k/10+pd.Series(probas_train4)*(10-i-j-k)/10
           probas_train_combine=probas_train_combine.tolist()
           auc_train = round(roc_auc_score(y_train, probas_train_combine),3)
           print(auc_train)
           if auc_train>auc_temp:
                auc_temp=auc_train
                result_temp=[i/10,j/10,k/10,(10-i-j-k)/10]

#best ENS
probas_train_combine = probas_train1*0.2+probas_train2*0.5+probas_train3*0.3+probas_train4*0.1