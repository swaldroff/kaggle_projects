#%%
import pandas as pd
import collections as ct
import numpy as np
from sklearn import preprocessing, metrics, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
#%%
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

def preprocessText(df):
    dfNew = pd.DataFrame(columns=['newText','wordCount','sentenceLength','avgWordLength','punctuationCount'])
    signs = set(',.:;"?!')
    
    for doc in df.text:
        
        cd = {c:val for c, val in ct.Counter(doc).items() if c in signs}
        
        lsPuncCount = sum(cd.values())
        
        lsWordCount = len(doc.split())
        lsLength = len(doc) - lsPuncCount
        avgWordLength = round(float(lsLength)/float(lsWordCount),4)
        
        doc = doc.replace("' ", " ' ")
        doc = doc.replace(" '", " ' ")

        

        prods = set(doc) & signs
        for sign in prods:
            doc = doc.replace(sign, ' {} '.format(sign) )
            
        doc = doc.replace("  "," ")
        
        newRow = pd.Series({'newText':doc,'wordCount':lsWordCount,'sentenceLength':lsLength,'avgWordLength':avgWordLength,'punctuationCount':lsPuncCount})
        
        dfNew = dfNew.append(newRow,ignore_index=True)
        
    return dfNew
    



#%%
dfTrain = pd.read_csv("D:/data_science/kaggle/spooky_authors/train.csv")

dfTrain.groupby('author')['author'].count()

lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(dfTrain.author.values)

#%%

xtrain, xvalid, ytrain, yvalid = train_test_split(dfTrain.text.values,y,stratify=y,random_state=19,test_size=0.3)


xtrain = preprocessText(pd.DataFrame(xtrain,columns=['text']))
xvalid = preprocessText(pd.DataFrame(xvalid,columns=['text']))

#%%
tfidf_vec = TfidfVectorizer(min_df=.0003,  max_features=None,strip_accents='unicode', analyzer='word', ngram_range=(1, 1),  use_idf=False, smooth_idf=1, sublinear_tf=1)

tfidf_vec.fit(list(xtrain.newText) + list(xvalid.newText))

xtrain_tfv = tfidf_vec.transform(xtrain.newText)
xvalid_tfv = tfidf_vec.transform(xvalid.newText)

xtrain_tfv = pd.DataFrame(xtrain_tfv.toarray(),columns = tfidf_vec.get_feature_names())
xvalid_tfv = pd.DataFrame(xvalid_tfv.toarray(),columns = tfidf_vec.get_feature_names())

#%%
count_vec = CountVectorizer(strip_accents='unicode',analyzer='word', lowercase = True, max_features = None, min_df = 0.0003)

count_vec.fit(list(xtrain.newText) + list(xvalid.newText))

xtrain_cv = count_vec.transform(xtrain.newText)
xvalid_cv = count_vec.transform(xvalid.newText)

xtrain_cv = pd.DataFrame(xtrain_cv.toarray(),columns = count_vec.get_feature_names())
xvalid_cv = pd.DataFrame(xvalid_cv.toarray(),columns = count_vec.get_feature_names())

#%%
# Fitting a simple Logistic Regression on TFIDF features only
lr = LogisticRegression(C=9.8999999999999932)
lr.fit(xtrain_tfv, ytrain)
predictions = lr.predict_proba(xvalid_tfv)

print(lr.score(xvalid_tfv, yvalid))

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


#%%
# Fitting a simple Logistic Regression on count_vec features only
lr = LogisticRegression(C = 1.0)
lr.fit(xtrain_cv,ytrain)
predictions = lr.predict_proba(xvalid_cv)

print(lr.score(xvalid_cv,yvalid))
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#%%
pieces = [xtrain_tfv,xtrain.iloc[:,1:4]]
xtrain_full = pd.concat(pieces,axis=1)
pieces = [xvalid_tfv,xvalid.iloc[:,1:4]]
xvalid_full = pd.concat(pieces,axis=1)

#%%
# Fitting a simple Logistic Regression on TFIDF + added features
lr = LogisticRegression(C=9.8999999999999932)
lr.fit(xtrain_full, ytrain)
predictions = lr.predict_proba(xvalid_full)
pred_class = lr.predict(xvalid_full)
print(lr.score(xvalid_full, yvalid))
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print (confusion_matrix(yvalid,pred_class))

model = lr

#%%
pieces = [xtrain_cv,xtrain.iloc[:,1:4]]
xtrain_full = pd.concat(pieces,axis=1)
pieces = [xvalid_cv,xvalid.iloc[:,1:4]]
xvalid_full = pd.concat(pieces,axis=1)

#%%
# Fitting a simple Logistic Regression on cv + added features
lr = LogisticRegression(C=1.0)
lr.fit(xtrain_full, ytrain)
predictions = lr.predict_proba(xvalid_full)
pred_class = lr.predict(xvalid_full)
print(lr.score(xvalid_full, yvalid))
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print (confusion_matrix(yvalid,pred_class))

model = lr

#%%
print(lr.coef_)
#%%

#https://stackoverflow.com/questions/39745807/typeerror-expected-sequence-or-array-like-got-estimator
xtrain_full_copy = xtrain_full.rename(columns = {'fit': 'fit_feature'})

scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
model_lr = LogisticRegression()
pl = pipeline.Pipeline([('lr',model_lr)])

param_grid = {'lr__C':np.arange(0.1, 2.0, .1)}

model = GridSearchCV(estimator=pl,param_grid=param_grid,scoring=scorer,verbose=100,cv=2)

model.fit(xtrain_full_copy,ytrain)

print(model.best_estimator_)

predictions = model.predict_proba(xvalid_full)
pred_class = model.predict(xvalid_full)
print(model.score(xvalid_full, yvalid))
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print (confusion_matrix(yvalid,pred_class))

#%%
model_mnb = MultinomialNB(alpha=.007)
model_mnb.fit(xtrain_full_copy,ytrain)

predictions = model_mnb.predict_proba(xvalid_full)
pred_class = model_mnb.predict(xvalid_full)
print(model_mnb.score(xvalid_full, yvalid))
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print (confusion_matrix(yvalid,pred_class))
#%%
#Neural Net

scaler = StandardScaler()

scaler.fit(xtrain_full_copy)

xtrain_mlp = scaler.transform(xtrain_full_copy)

model_mlp = MLPClassifier(alpha=0.00001,hidden_layer_sizes=(40,40,),activation='relu',solver='adam', random_state=19,verbose=True,max_iter=100,tol=0.00000001)
model_mlp.fit(xtrain_mlp,ytrain)

xvalid_mlp = scaler.transform(xvalid_full)

predictions = model_mlp.predict_proba(xvalid_mlp)
pred_class = model_mlp.predict(xvalid_mlp)
print(model_mlp.score(xvalid_mlp, yvalid))
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print (confusion_matrix(yvalid,pred_class))


#%%
#Prepare and run on test dataset
dfTest = pd.read_csv("D:\\data_science\\kaggle\\spooky_authors\\test.csv")

dfSubmission = pd.read_csv("D:\\data_science\\kaggle\\spooky_authors\\submissions\\sample_submission.csv")

xtest = preprocessText(dfTest)
xtest_tfv = tfidf_vec.transform(xtest.newText)
xtest_tfv = pd.DataFrame(xtest_tfv.toarray(),columns = tfidf_vec.get_feature_names())
pieces = [xtest_tfv,xtest.iloc[:,1:4]]
xtest_full = pd.concat(pieces,axis=1)


sub_predictions = pd.DataFrame(model.predict_proba(xtest_full),columns=['EAP','HPL','MWS'])

dfSubmission['EAP'] = sub_predictions['EAP']
dfSubmission['HPL'] = sub_predictions['HPL']
dfSubmission['MWS'] = sub_predictions['MWS']


dfSubmission.to_csv("D:\\data_science\\kaggle\\spooky_authors\\submissions\\sub_test5.csv",index=False)


#%%




