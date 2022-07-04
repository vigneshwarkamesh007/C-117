import statistics as st
import pandas as pd
import plotly.express as pe
import plotly.figure_factory as pf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.metrics import confusion_matrix
data= pd.read_csv("bank.csv")
variance = data["variance"]
classvar = data["class"]

vartrain, vartest, classvartrain, classvartest = train_test_split(variance, classvar, test_size=0.25, random_state=0)

#Training the data
Xtrain=np.reshape(vartrain.ravel(), (len(vartrain), 1))
Ytrain=np.reshape(classvartrain.ravel(), (len(classvartrain), 1))
classifier = LogisticRegression(random_state=0)
classifier.fit(Xtrain,Ytrain)

#Predicted Value, and testing
Xtest = np.reshape(vartest.ravel(), (len(vartest),1))
Ytest = np.reshape(classvartest.ravel(), (len(classvartest),1))
stealprediction = classifier.predict(classvartest)
predictedvalues = []
actualvalues=[]

for i in stealprediction:
    if(stealprediction == 0):
        predictedvalues.append("Authorized")
    else:
        predictedvalues.append("Forged")

for i in Ytest:
  if(stealprediction == 0):
        actualvalues.append("Authorized")
  else:
        actualvalues.append("Forged")


labels=["Yes", "No"]
confusionmatrix= confusion_matrix(actualvalues,predictedvalues, labels)
ax=mp.subplot()
sb.heatmap(confusionmatrix, annot=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

