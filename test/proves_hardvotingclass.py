from __future__ import print_function, division
from collections import Counter
import os



def hardVotingClassifier(preds_model1, preds_model2, preds_model3):
    """Se li passen els 4 tensors amb la info acumulada. S'ha de fer la moda de cada imatge"""
    hard_voting_preds=[]
    for i in range(len(preds_model1)):
        print('i',i)
        list_preds = [preds_model1[i], preds_model2[i], preds_model3[i]]  # llista amb les tres posicions
        print(list_preds)
        hard_voting_preds.append(max(set(list_preds), key=list_preds.count))
        print(hard_voting_preds)

    return hard_voting_preds
a=[0,1,2,3]
b=[0,2,2,4]
c=[0,3,1,4]

return_value=hardVotingClassifier(a,b,c)
print('return_value',return_value)


#preds train model1:  tensor([1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0,2, 3, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
#preds train model2:  tensor([1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#preds train model3:  tensor([1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
#preds_ensemble: [tensor(1), tensor(0), tensor(0), tensor(3),
# tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(1), tensor(0), tensor(0), tensor(0), tensor(0), t
ensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0),
tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0), tensor(0
), tensor(0), tensor(0)]
