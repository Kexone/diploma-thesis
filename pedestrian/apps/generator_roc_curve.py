import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import glob
import os

y_test = np.loadtxt('../data/tested/test.txt', delimiter=' ', skiprows=1 )
rocFiles = [f for f in glob.glob("../data/tested/*_test.txt")]
rocList = []
#print (y_test)
print (len(rocFiles))

for i in range(0, len(rocFiles)):
    predicted = np.loadtxt(rocFiles[i], delimiter=' ', skiprows=1, usecols=1)
    fpr, tpr, _ = roc_curve(y_test, predicted)
    roc = auc(fpr, tpr)
    rocList.append({
        "name" : "test",
        "roc" : roc,
        "fpr" : fpr,
        "tpr" : tpr
        })
rocList.sort(key=lambda c: c["roc"], reverse=True)

for i in range(0, len(rocList)):
    print ("{0} ROC AUC: {1}".format("blabla", round(rocList[i]["roc"],3)))

#plt.rcParams.update({'font.size: 14'})
                     
plt.figure()
for i in range(0, len(rocList)):
    plt.plot(rocList[i]["fpr"], rocList[i]["tpr"], label="${0} : {1}$".format("blabla",
                                                                              round(rocList[i]["roc"],3)),
             linewidth=2.2)        
plt.plot([0, 1], [0, 1], 'r')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
