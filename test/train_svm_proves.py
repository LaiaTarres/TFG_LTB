import sys
import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss

from sklearn.svm import SVC




model=sys.argv[1]
print('se li ha de passar el model sense pth [output_model_my_resnet_18_2018]', model)
#print('srun --mem 16G python train_svm_proves.py output_model_my_resnet_18_2018')

labels_train_file='/imatge/ltarres/PycharmProjects/test/model_output/train_labels_'+model+'.txt'
params_train_file='/imatge/ltarres/PycharmProjects/test/model_output/train_params_'+model+'.txt'
preds_train_file='/imatge/ltarres/PycharmProjects/test/model_output/train_preds_'+model+'.txt'

labels_val_file='/imatge/ltarres/PycharmProjects/test/model_output/val_labels_'+model+'.txt'
params_val_file='/imatge/ltarres/PycharmProjects/test/model_output/val_params_'+model+'.txt'
preds_val_file='/imatge/ltarres/PycharmProjects/test/model_output/val_preds_'+model+'.txt'


with open(labels_train_file, 'r') as textFile: #ja esta be
  labels1t = textFile.readlines()
labels_train = [float(s.replace('\n', '')) for s in labels1t]

with open(params_train_file, 'r') as textFile:
    params1t = textFile.readlines()
params_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1t]

with open(preds_train_file, 'r') as textFile: #ja està bé
  preds1t = textFile.readlines()
preds_train = [float(s.replace('\n', '')) for s in preds1t]



with open(labels_val_file, 'r') as textFile:
  labels1v = textFile.readlines()
labels_val = [float(s.replace('\n', '')) for s in labels1v]

with open(params_val_file, 'r') as textFile:
    params1v = textFile.readlines()
params_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1v]

with open(preds_val_file, 'r') as textFile:
  preds1v = textFile.readlines()
preds_val = [float(s.replace('\n', '')) for s in preds1v]




def svc_param_selection(X, y, nfolds):
     tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1e-3, 1e-4, 1e-5],
                          'C': [ 1, 10, 25, 50, 100, 1000]},
                         #{'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         # 'C': [0.001, 0.01, 0.1, 10, 25, 50, 100, 1000]},
                         #{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]}
                         ]
     print('tuned_parameters',tuned_parameters)
     grid_search = GridSearchCV(SVC(C=1.0), tuned_parameters, cv=nfolds, n_jobs=-1, verbose=10)
     grid_search.fit(X, y)
     means = grid_search.cv_results_['mean_test_score']
     for mean, params in zip(means, grid_search.cv_results_['params']):
         print("%0.3f for %r"% (mean, params))
     grid_search.best_params_
     return grid_search.best_params_

print('Els millors paràmetres probats al gridsearch son:',svc_param_selection(params_train, labels_train, 3))

#Create a svm Classifier
#clf = svm.SVC(C=1.0,kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0) # Linear Kernel
#Train the model using the training sets
#clf.fit(params_train, labels_train) #X_train, Y_train
#Predict the response for test dataset
#preds_svm = clf.predict(params_val) #y_pred = clf.predict(X_test)
#to evaluate the svm model
#Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(preds_val, preds_svm))#y_test, y_pred, per comparar el que hauria de ser amb el que és

#Així d'entrada, l'accuracy és de 0.215676
#per millorar-ho: kernel,regularization, gamma
#kernel: opcions: 'linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
#linear: K(x, xi) = sum(x * xi)
#polynomial: K(x,xi) = 1 + sum(x * xi)^d, s'ha d'expecificar quina d s'utilitza
#radial basis function: K(x,xi) = exp(-gamma * sum((x – xi^2)), d'entrada diuen que provem amb gamma=0.1

#regularization: la C penalitza l'error màxim que es pot permetre en training. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
#clf = svm.SVC(C=0.9,kernel='linear') # Linear Kernel

#gamma: A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.
#

#En realitat els paràmetres que es poden modificar son:
#SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)

#amb linear i C=0.5 -> 0.21917
#amb polynomial de grau 3 tenim i C=0.5 -> 0.1323
#amb polynomial de grau 3 tenim i C=1.0 -> 0.1467
#amb rbf de grau 3 gamma=auto_deprecated, coef0=0.0 -> 0.1707
#

#creem un svm per cada classificador amb els millors paràmetres trobats a dalt.
#els millors coefs son:
