# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:06:09 2019

@author: arodriguez
"""

import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools

from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
import seaborn
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV



train = pd.read_csv("credit_train.csv") 
test = pd.read_csv("credit_test.csv") 


'''


IMPORTANT: sOMETIMES YOU NEED TO RUN THE CODE UNTIL THE IMBALANCE SECTION, 
THE ITERATIVE IMPUTER, BECAUSE IF YOU TRY TO RUN THE WHOLE CODE AT ONCE IT GIVES AN ERROR
SO BETTER RUN THE CODE UNTIL THE IMBALANCE ITERATIVE IMPUTER AND THEN YOU CAN RUN
THE ENTIRE CODE


    
    
    '''

'''PREPROCESSING'''


''' Dealing with outliers'''

sns.boxplot(data=train)
plt.show

Q1 = train.iloc[:,16].quantile(0.25)
Q3 = train.iloc[:,16].quantile(0.75)
IQR = Q3 - Q1

train = train.loc[~((train.iloc[:,16] < (Q1 - 3 * IQR)) |(train.iloc[:,16] > (Q3 + 3 * IQR)))]
train.shape


Q1 = train.iloc[:,3].quantile(0.25)
Q3 = train.iloc[:,3].quantile(0.75)
IQR = Q3 - Q1

train = train.loc[~((train.iloc[:,3] < (Q1 - 3.5 * IQR)) |(train.iloc[:,3] > (Q3 + 3.5 * IQR)))]
train.shape

Q1 = train.iloc[:,6].quantile(0.25)
Q3 = train.iloc[:,6].quantile(0.75)
IQR = Q3 - Q1

train = train.loc[~((train.iloc[:,6] < (Q1 - 3.5 * IQR)) |(train.iloc[:,6] > (Q3 + 3.5 * IQR)))]
train.shape

''' Dealing with missing values'''

train=train.drop(['Months since last delinquent'], axis=1)
train=train.drop(['Loan Status'], axis=1)

train = train.loc[~(train.iloc[:,0].isna())]
train.shape
print(train.isnull().sum())

train_miss=train.copy()
'''train_miss=train_miss.drop(['Loan ID','Customer ID'], axis=1)'''

train_miss['Years in current job'] = train_miss['Years in current job'].astype(str)

train_miss.reset_index(drop=True, inplace=True)
train_miss.index = pd.to_numeric(train_miss.index, errors='coerce')





'''train_miss_np=train_miss.to_numpy()'''

from sklearn import preprocessing



le2 = preprocessing.LabelEncoder()
le2.fit(train_miss['Term'])
b=le2.transform(train_miss['Term'])
train_miss['Term'] =le2.transform(train_miss['Term'])

le3 = preprocessing.LabelEncoder()
le3.fit(train_miss['Years in current job'])
c=le3.transform(train_miss['Years in current job'])
train_miss['Years in current job'] =le3.transform(train_miss['Years in current job'])

le4 = preprocessing.LabelEncoder()
le4.fit(train_miss['Home Ownership'])
d=le4.transform(train_miss['Home Ownership'])
train_miss['Home Ownership'] =le4.transform(train_miss['Home Ownership'])


le5 = preprocessing.LabelEncoder()
le5.fit(train_miss['Purpose'])
e=le5.transform(train_miss['Purpose'])
train_miss['Purpose'] =le5.transform(train_miss['Purpose'])


train_miss=train_miss.drop(['Loan ID','Customer ID'],axis=1)


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp=IterativeImputer(random_state=2)
imp.fit(train_miss)
res_imp=imp.transform(train_miss)



train_miss['Term']=le2.inverse_transform(b)
train_miss['Years in current job']=le3.inverse_transform(c) 
train_miss['Home Ownership']=le4.inverse_transform(d)
train_miss['Purpose']=le5.inverse_transform(e)


train_miss['Credit Score']=res_imp[:,2]
train_miss['Annual Income']=res_imp[:,3]
train_miss['Maximum Open Credit']=res_imp[:,12]
train_miss['Bankruptcies']=res_imp[:,13]
train_miss['Tax Liens']=res_imp[:,14]

train_miss.replace('nan', np.nan, inplace=True)


train_miss=train_miss.dropna()




'''Handling Categorical Data'''


train_miss['Term'] =le2.transform(train_miss['Term'])

train_miss['Years in current job'] =le3.transform(train_miss['Years in current job'])

train_miss['Home Ownership'] =le4.transform(train_miss['Home Ownership'])

train_miss['Purpose'] =le5.transform(train_miss['Purpose'])


train_miss['Bankruptcies'][train_miss['Bankruptcies'] < 0 ] = 0
train_miss['Bankruptcies'][ train_miss['Bankruptcies'].between(0,1,inclusive=False)] = 0
train_miss['Bankruptcies'][ train_miss['Bankruptcies'].between(2,7,inclusive=True)] = 0

train_miss['Tax Liens'][train_miss['Tax Liens'] < 0 ] = 0
train_miss['Tax Liens'][ train_miss['Tax Liens'].between(0,1,inclusive=False)] = 0


'''Scaling Data'''

scaler1=preprocessing.StandardScaler()
train_miss['Current Loan Amount']=scaler1.fit_transform(train_miss['Current Loan Amount'].values.reshape(-1,1))

scaler2=preprocessing.StandardScaler()
train_miss['Credit Score']=scaler2.fit_transform(train_miss['Credit Score'].values.reshape(-1,1))

scaler3=preprocessing.StandardScaler()
train_miss['Annual Income']=scaler3.fit_transform(train_miss['Annual Income'].values.reshape(-1,1))

scaler4=preprocessing.StandardScaler()
train_miss['Monthly Debt']=scaler4.fit_transform(train_miss['Monthly Debt'].values.reshape(-1,1))

scaler5=preprocessing.StandardScaler()
train_miss['Years of Credit History']=scaler5.fit_transform(train_miss['Years of Credit History'].values.reshape(-1,1))

scaler6=preprocessing.StandardScaler()
train_miss['Number of Open Accounts']=scaler6.fit_transform(train_miss['Number of Open Accounts'].values.reshape(-1,1))

scaler7=preprocessing.StandardScaler()
train_miss['Current Credit Balance']=scaler7.fit_transform(train_miss['Current Credit Balance'].values.reshape(-1,1))

scaler8=preprocessing.StandardScaler()
train_miss['Maximum Open Credit']=scaler8.fit_transform(train_miss['Maximum Open Credit'].values.reshape(-1,1))

scalingObj1=preprocessing.MinMaxScaler()
train_miss['Years in current job']=scalingObj1.fit_transform(train_miss['Years in current job'].values.reshape(-1,1))

scalingObj2=preprocessing.MinMaxScaler()
train_miss['Home Ownership']=scalingObj2.fit_transform(train_miss['Home Ownership'].values.reshape(-1,1))

scalingObj3=preprocessing.MinMaxScaler()
train_miss['Purpose']=scalingObj3.fit_transform(train_miss['Purpose'].values.reshape(-1,1))

scalingObj4=preprocessing.MinMaxScaler()
train_miss['Number of Credit Problems']=scalingObj4.fit_transform(train_miss['Number of Credit Problems'].values.reshape(-1,1))

scalingObj6=preprocessing.MinMaxScaler()
train_miss['Tax Liens']=scalingObj6.fit_transform(train_miss['Tax Liens'].values.reshape(-1,1))




'''IMBALANCE SECTION'''



'''imbalanced data'''
'''import imblearn'''
from imblearn.over_sampling import SMOTE

X=np.array(train_miss.iloc[:,train_miss.columns != 'Bankruptcies'])
y=np.array(train_miss.iloc[:,train_miss.columns == 'Bankruptcies'])

print("Before OverSampling, counts of label '1': {}".format(sum(y==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y==0)))

sm=SMOTE(sampling_strategy=1,random_state=1)
X_train_res, y_train_res = sm.fit_sample(X,y.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

'''Feature Selection'''
train_miss_f=pd.DataFrame()
i=-1
for column in train_miss:
    if column == 'Bankruptcies':
        column = 'Tax Liens'
    train_miss_f[column]=X_train_res[:,i]
    i=i+1

from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
selector=SelectPercentile(f_regression,percentile=10)
selector.fit(X_train_res,y_train_res)

for n,s in zip(train_miss_f,selector.scores_):
    print("Score:",s,"for feature",n)
    
    
    
    

'''TEST PREPROCESSING'''



test_miss=test.copy()
test_miss=test_miss.drop(['Months since last delinquent'], axis=1)

'''train_miss=train_miss.drop(['Loan ID','Customer ID'], axis=1)'''

test_miss['Years in current job'] = test_miss['Years in current job'].astype(str)
test_miss['Term'] = test_miss['Term'].astype(str)
test_miss['Home Ownership'] = test_miss['Home Ownership'].astype(str)
test_miss['Purpose'] = test_miss['Purpose'].astype(str)

test_miss.reset_index(drop=True, inplace=True)
test_miss.index = pd.to_numeric(test_miss.index, errors='coerce')





'''train_miss_np=train_miss.to_numpy()'''

from sklearn import preprocessing



le20 = preprocessing.LabelEncoder()
le20.fit(test_miss['Term'])
b=le20.transform(test_miss['Term'])
test_miss['Term'] =le20.transform(test_miss['Term'])

le30 = preprocessing.LabelEncoder()
le30.fit(test_miss['Years in current job'])
c=le30.transform(test_miss['Years in current job'])
test_miss['Years in current job'] =le30.transform(test_miss['Years in current job'])

le40 = preprocessing.LabelEncoder()
le40.fit(test_miss['Home Ownership'])
d=le40.transform(test_miss['Home Ownership'])
test_miss['Home Ownership'] =le40.transform(test_miss['Home Ownership'])


le50 = preprocessing.LabelEncoder()
le50.fit(test_miss['Purpose'])
e=le50.transform(test_miss['Purpose'])
test_miss['Purpose'] =le50.transform(test_miss['Purpose'])


test_miss=test_miss.drop(['Loan ID','Customer ID'],axis=1)


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp1=IterativeImputer(random_state=2)
imp1.fit(test_miss)
res_imp1=imp1.transform(test_miss)



test_miss['Term']=le20.inverse_transform(b)
test_miss['Years in current job']=le30.inverse_transform(c) 
test_miss['Home Ownership']=le40.inverse_transform(d)
test_miss['Purpose']=le50.inverse_transform(e)


test_miss['Credit Score']=res_imp1[:,2]
test_miss['Annual Income']=res_imp1[:,3]
test_miss['Maximum Open Credit']=res_imp1[:,12]
test_miss['Bankruptcies']=res_imp1[:,13]
test_miss['Tax Liens']=res_imp1[:,14]

test_miss.replace('nan', np.nan, inplace=True)


test_miss=test_miss.dropna()




'''Handling Categorical Data'''


test_miss['Term'] =le20.transform(test_miss['Term'])

test_miss['Years in current job'] =le30.transform(test_miss['Years in current job'])

test_miss['Home Ownership'] =le40.transform(test_miss['Home Ownership'])

test_miss['Purpose'] =le50.transform(test_miss['Purpose'])


test_miss['Bankruptcies'][test_miss['Bankruptcies'] < 0 ] = 0
test_miss['Bankruptcies'][ test_miss['Bankruptcies'].between(0,1,inclusive=False)] = 0
test_miss['Bankruptcies'][ test_miss['Bankruptcies'].between(2,7,inclusive=True)] = 0

test_miss['Tax Liens'][test_miss['Tax Liens'] < 0 ] = 0
test_miss['Tax Liens'][ test_miss['Tax Liens'].between(0,1,inclusive=False)] = 0


'''Scaling Data'''


test_miss['Current Loan Amount']=scaler1.fit_transform(test_miss['Current Loan Amount'].values.reshape(-1,1))

test_miss['Credit Score']=scaler2.fit_transform(test_miss['Credit Score'].values.reshape(-1,1))

test_miss['Annual Income']=scaler3.fit_transform(test_miss['Annual Income'].values.reshape(-1,1))


test_miss['Monthly Debt']=scaler4.fit_transform(test_miss['Monthly Debt'].values.reshape(-1,1))

test_miss['Years of Credit History']=scaler5.fit_transform(test_miss['Years of Credit History'].values.reshape(-1,1))

test_miss['Number of Open Accounts']=scaler6.fit_transform(test_miss['Number of Open Accounts'].values.reshape(-1,1))

test_miss['Current Credit Balance']=scaler7.fit_transform(test_miss['Current Credit Balance'].values.reshape(-1,1))

test_miss['Maximum Open Credit']=scaler8.fit_transform(test_miss['Maximum Open Credit'].values.reshape(-1,1))

test_miss['Years in current job']=scalingObj1.fit_transform(test_miss['Years in current job'].values.reshape(-1,1))

test_miss['Home Ownership']=scalingObj2.fit_transform(test_miss['Home Ownership'].values.reshape(-1,1))

test_miss['Purpose']=scalingObj3.fit_transform(test_miss['Purpose'].values.reshape(-1,1))

test_miss['Number of Credit Problems']=scalingObj4.fit_transform(test_miss['Number of Credit Problems'].values.reshape(-1,1))

test_miss['Tax Liens']=scalingObj6.fit_transform(test_miss['Tax Liens'].values.reshape(-1,1))



'''function that returns a confusion matrix. The code is taken from this source: 
    https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd'''
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
    
    
    
    
    
    
    
    
'''Section 3 of the Paper (Methodolody) testing all models in the list'''

'''Logistic Regression CV'''

from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=5,random_state=1).fit(X_train_res,y_train_res)

from sklearn.metrics import accuracy_score

X_test=np.array(test_miss.iloc[:,test_miss.columns != 'Bankruptcies'])
y_test=np.array(test_miss.iloc[:,test_miss.columns == 'Bankruptcies'])

predictions=clf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(acc)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools

from sklearn.metrics import precision_recall_fscore_support
ypredict=clf.predict(X_test)
print(precision_recall_fscore_support(y_test, ypredict))

cnf_matrix = metrics.confusion_matrix(y_test, ypredict)
cnf_matrix



plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0','1'],
                      title='Confusion matrix')


'''SVM'''
from sklearn import svm
import seaborn

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train_res, y_train_res) 
prediction_SVM_all = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_test, prediction_SVM_all)
plot_confusion_matrix(cm, classes=['0','1'],
                      title='Confusion matrix')


'''SGD Classifier'''

from sklearn.linear_model import SGDClassifier

model_hinge = SGDClassifier(loss = 'hinge', penalty = 'l2')
model_hinge = model_hinge.fit(X_train_res, y_train_res)
ypred = model_hinge.predict(X_test)
avg = 0.0
n = 200
for i in range(n):
    avg = avg + model_hinge.score(X_test,y_test)
print(avg / n)


cm2 = metrics.confusion_matrix(y_test, ypred)
plot_confusion_matrix(cm2, classes=['0','1'],
                      title='Confusion matrix')

'''Decission Tree'''
from sklearn import tree
dt1 = tree.DecisionTreeClassifier()
dt1.fit(X_train_res, y_train_res)
dt1_score_train = dt1.score(X_train_res, y_train_res)
print("Training score: ", dt1_score_train)
dt1_score_test = dt1.score(X_test,y_test)
print("Testing score: ", dt1_score_test)

pred_tree=dt1.predict(X_test)

cm3 = metrics.confusion_matrix(y_test, pred_tree)
plot_confusion_matrix(cm3, classes=['0','1'],
                      title='Confusion matrix')


'''Random Forest'''

from sklearn.ensemble import RandomForestClassifier
rclf = RandomForestClassifier()
rclf.fit(X_train_res,y_train_res)

pred_RF=rclf.predict(X_test)

cm4 = metrics.confusion_matrix(y_test, pred_RF)
plot_confusion_matrix(cm4, classes=['0','1'],
                      title='Confusion matrix')


'''Ada Boost'''

from sklearn.ensemble import AdaBoostClassifier
Aclf = AdaBoostClassifier()
Aclf.fit(X_train_res,y_train_res)

pred_AB=Aclf.predict(X_test)

cm5 = metrics.confusion_matrix(y_test, pred_AB)
plot_confusion_matrix(cm5, classes=['0','1'],
                      title='Confusion matrix')

'''Gradient Boosting'''

from sklearn.ensemble import GradientBoostingClassifier
GBclf = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=150, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
GBclf.fit(X_train_res,y_train_res)

pred_GB=GBclf.predict(X_test)

cm6 = metrics.confusion_matrix(y_test, pred_GB)
plot_confusion_matrix(cm6, classes=['0','1'],
                      title='Confusion matrix')


'''Neural Networks'''

from sklearn.neural_network import MLPClassifier

NNclf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)

NNclf.fit(X_train_res, y_train_res)

pred_NN=NNclf.predict(X_test)

cm7 = metrics.confusion_matrix(y_test, pred_NN)
plot_confusion_matrix(cm7, classes=['0','1'],
                      title='Confusion matrix')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res)
pred_LR=lr.predict(X_test)

cm8 = metrics.confusion_matrix(y_test, pred_LR)
plot_confusion_matrix(cm8, classes=['0','1'],
                      title='Confusion matrix')










'''From the previous models we select the favorite models:
    Logistic Regression, SVM, Random Forest and Gradient Boosting'''
    
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

'''Logistic Regression'''
gridlr={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
lrCV=GridSearchCV(lr,gridlr,cv=10)
lrCV.fit(X_train_res, y_train_res)

pred_LR_CV=lrCV.predict(X_test)

cm_cv1 = metrics.confusion_matrix(y_test, pred_LR_CV)
plot_confusion_matrix(cm_cv1, classes=['0','1'],
                      title='Confusion matrix')

'''SVM'''
svmCV=svm.SVC()
gridSVM = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
SVMCV = GridSearchCV(svmCV, gridSVM)
SVMCV.fit(X_train_res, y_train_res)

pred_SVM_CV=SVMCV.predict(X_test)

cm_cv2 = metrics.confusion_matrix(y_test, pred_SVM_CV)
plot_confusion_matrix(cm_cv2, classes=['0','1'],
                      title='Confusion matrix')

from sklearn.ensemble import RandomForestClassifier
'''Random Forest'''

gridRF = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['entropy']
}

CV_rfc = GridSearchCV(rclf, gridRF, n_jobs=-1)
CV_rfc.fit(X_train_res, y_train_res)

pred_RF_CV=CV_rfc.predict(X_test)

cm_cv3 = metrics.confusion_matrix(y_test, pred_RF_CV)
plot_confusion_matrix(cm_cv3, classes=['0','1'],
                      title='Confusion matrix')


''' Gradient Boosting'''
gridGB = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

GBCV = GridSearchCV(GBclf, gridGB, cv=10, n_jobs=-1)
GBCV.fit(X_train_res, y_train_res)

pred_GB_CV=GBCV.predict(X_test)

cm_cv4 = metrics.confusion_matrix(y_test, pred_GB_CV)
plot_confusion_matrix(cm_cv4, classes=['0','1'],
                      title='Confusion matrix')








''' Research Section of the Paper (Section 2): Problem with False Positives'''

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


'''We use several variants of SMOTE as the paper mentions: This is taken from:
    https://pypi.org/project/smote-variants/#id2'''



import smote_variants as sv
'''SMOTE_Tomeklinks'''
oversampler= sv.SMOTE_TomekLinks(random_state=1)

X_samp, y_samp= oversampler.sample(X, y.ravel())


'''Random Forest'''

rclf = RandomForestClassifier()
rclf.fit(X_samp,y_samp)

pred_RF_SL=rclf.predict(X_test)

cm4_RF_SL = metrics.confusion_matrix(y_test, pred_RF_SL)
plot_confusion_matrix(cm4_RF_SL, classes=['0','1'],
                      title='Confusion matrix')


'''SMOTE_ENN'''
oversampler2= sv.SMOTE_ENN(random_state=1)

X_samp, y_samp= oversampler2.sample(X, y.ravel())

'''Random Forest'''

rclf.fit(X_samp,y_samp)

pred_RF_ENN=rclf.predict(X_test)

cm4_RF_ENN = metrics.confusion_matrix(y_test, pred_RF_ENN)
plot_confusion_matrix(cm4_RF_ENN, classes=['0','1'],
                      title='Confusion matrix')


'''SVM_balance'''
oversampler3= sv.SVM_balance(random_state=1)

X_samp, y_samp= oversampler3.sample(X, y.ravel())

'''Random Forest'''

rclf.fit(X_samp,y_samp)

pred_RF_SVM_balance=rclf.predict(X_test)

cm4_RF_SVM_Balance = metrics.confusion_matrix(y_test, pred_RF_SVM_balance)
plot_confusion_matrix(cm4_RF_SVM_Balance, classes=['0','1'],
                      title='Confusion matrix')

'''SVM_balance'''
oversampler4= sv.ADASYN(random_state=1)

X_samp, y_samp= oversampler4.sample(X, y.ravel())

'''Random Forest'''

rclf.fit(X_samp,y_samp)

pred_RF_ADASYN=rclf.predict(X_test)

cm4_RF_ADASYN = metrics.confusion_matrix(y_test, pred_RF_ADASYN)
plot_confusion_matrix(cm4_RF_ADASYN, classes=['0','1'],
                      title='Confusion matrix')

'''ISOMAP_hybrid'''
oversampler5= sv.ISOMAP_Hybrid(random_state=1)

X_samp, y_samp= oversampler5.sample(X, y.ravel())

'''Random Forest'''

rclf.fit(X_samp,y_samp)

pred_RF_ISO_Hybrid=rclf.predict(X_test)

cm4_RF_ISO_Hybrid = metrics.confusion_matrix(y_test, pred_RF_ISO_Hybrid)
plot_confusion_matrix(cm4_RF_ISO_Hybrid, classes=['0','1'],
                      title='Confusion matrix')



'''Polyfit'''
oversampler6= sv.polynom_fit_SMOTE(random_state=1)

X_samp, y_samp= oversampler6.sample(X, y.ravel())

'''Random Forest'''

rclf.fit(X_samp,y_samp)

pred_RF_Poly=rclf.predict(X_test)

cm4_RF_Poly = metrics.confusion_matrix(y_test, pred_RF_Poly)
plot_confusion_matrix(cm4_RF_Poly, classes=['0','1'],
                      title='Confusion matrix')

'''CURE_SMOTE'''
oversampler7= sv.CURE_SMOTE(random_state=1)

X_samp, y_samp= oversampler7.sample(X, y.ravel())

'''Random Forest'''

rclf.fit(X_samp,y_samp)

pred_CURE_SMOTE=rclf.predict(X_test)

cm4_RF_CURE_SMOTE = metrics.confusion_matrix(y_test, pred_CURE_SMOTE)
plot_confusion_matrix(cm4_RF_CURE_SMOTE, classes=['0','1'],
                      title='Confusion matrix')









'''In this section we try to solve the problem of False Positves with Ensemble Learning
    Section 2.2 of the Paper'''

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
'''SVM'''
classifier = svm.SVC(kernel='linear',probability=True,random_state=1)

'''Gradient Tree'''
GBclf = GradientBoostingClassifier(random_state=1)

'''Logistic Regression'''
lr = LogisticRegression()

gridlr={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
lrCV=GridSearchCV(lr,gridlr,cv=10)
lrCV.fit(X_train_res, y_train_res)

pred_LR_CV=lrCV.predict(X_test)

cm_cv1 = metrics.confusion_matrix(y_test, pred_LR_CV)
plot_confusion_matrix(cm_cv1, classes=['0','1'],
                      title='Confusion matrix')


'''Neural Network: Not used for ensemble'''
NNclf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)

clf=LogisticRegressionCV(cv=5,random_state=1)



'''Voting Classifier with weights'''

voting_clf = VotingClassifier([("svm", classifier),
                               ("GB", GBclf),
                               ("lr", lrCV)],
                              voting="soft", weights=[0.13,0.74,0.13])
voting_clf.fit(X_train_res,y_train_res)
svm_model, GB_model, NN_model = voting_clf.estimators_
models = {"svm": svm_model,
          "GB": GB_model,
          "NN": NN_model,
          "avg_ensemble": voting_clf}

pred_voting_clf=voting_clf.predict(X_test)

cm4_RF_voting_clf = metrics.confusion_matrix(y_test, pred_voting_clf)
plot_confusion_matrix(cm4_RF_voting_clf, classes=['0','1'],
                      title='Confusion matrix')


'''voting_clf2 = VotingClassifier([("svm", svm_model),
                               ("GB", GB_model),
                               ("NN", NN_model)],voting='soft',weights=[1,2,2])



voting_clf2.fit(X_train_res,y_train_res)
pred_voting_clf2=voting_clf2.predict(X_test)'''


print(voting_clf.predict_proba)
from sklearn.model_selection import cross_val_score
labels = ['svm', 'GB', 'NN', 'Ensemble']
for clf, label in zip([svm_model, GB_model, NN_model, voting_clf], labels):

    scores = cross_val_score(clf, X_train_res, y_train_res, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    
    pred=clf.predict(X_test)

    cm = metrics.confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes=['0','1'],
                          title='Confusion matrix')