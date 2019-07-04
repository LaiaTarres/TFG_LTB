from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.metrics import balanced_accuracy_score


# Loading the data
model=sys.argv[1]
print('està fent main_sklearn_ensemble.py pel model : ',model)
labels_train_file='/imatge/ltarres/PycharmProjects/test/model_output/train_labels_'+model+'.txt' #els labels son tots iguals independentment del model
labels_val_file='/imatge/ltarres/PycharmProjects/test/model_output/val_labels_'+model+'.txt'

params_train_file='/imatge/ltarres/PycharmProjects/test/model_output/train_params_'+model+'.txt'
params_val_file='/imatge/ltarres/PycharmProjects/test/model_output/val_params_'+model+'.txt'

with open(labels_train_file, 'r') as textFile: #ja esta be
  labels1t = textFile.readlines()
labels_train = [float(s.replace('\n', '')) for s in labels1t]
with open(labels_val_file, 'r') as textFile:
  labels1v = textFile.readlines()
labels_val = [float(s.replace('\n', '')) for s in labels1v]
with open(params_train_file, 'r') as textFile:
    params1t = textFile.readlines()
params_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1t]
with open(params_val_file, 'r') as textFile:
    params1v = textFile.readlines()
params_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1v]

X=params_train
y=labels_train
W=params_val
z=labels_val



# Training classifiers

clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma=0.00001, kernel='rbf', C=1000, probability=True)
#de l'altre exemple
#clf4 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1) #em surt failed to converge, increase number of iterations
clf5 = RandomForestClassifier(n_estimators=50, random_state=1)
clf6 = GaussianNB()
#eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
#fi de l'altre exemple
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3), ('rf', clf5), ('gnb', clf6)], voting='hard') #, weights=[2, 1, 2, 1, 2]
print('El ensmeble utilizado es de tipo: hard sin pesos')
clf1 = clf1.fit(X,y)
clf2 = clf2.fit(X,y)
clf3 = clf3.fit(X,y)
clf5 = clf5.fit(X,y)
clf6 = clf6.fit(X,y)
eclf = eclf.fit(X,y)
#s'hauria de fer que les prediccions es guardessin en un fitxer...

#M'està donant l'accuracy amb els valors de training, també ho he de fer amb els de val
for clf, label in zip([clf1, clf2, clf3, clf5, clf6, eclf], ['Decision Tree','KNeighbors','SVC', 'Random Forest', 'naive Bayes',  'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='balanced_accuracy')
    print("Balanced Accuracy training: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    preds_train = clf.predict(X)
    my_balanced_acc = balanced_accuracy_score(y, preds_train, adjusted=True)
    print('My Balanced Accuracy val: %0.2f' % my_balanced_acc)



for clf, label in zip([clf1, clf2, clf3, clf5, clf6, eclf], ['Decision Tree','KNeighbors','SVC', 'Random Forest', 'naive Bayes',  'Ensemble']):
    scores = cross_val_score(clf, W, z, cv=5, scoring='balanced_accuracy')
    print("Balanced Accuracy val: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    preds_val=clf.predict(W)

    my_balanced_acc = balanced_accuracy_score(z,preds_val , adjusted=True)
    print('My Balanced Accuracy val: %0.2f' %my_balanced_acc)


##Després de fer el grid search, el svm que ens ha donat millors resultats ha estat: