from __future__ import print_function, division
from collections import counter

def HardVotingClassifier(preds_model1, preds_model2, preds_model3):
    """Se li passen els 4 tensors amb la info acumulada. S'ha de fer la moda de cada imatge"""

    for i in int(preds_model1.size):
        list_preds=[preds_model1[i],preds_model1[i],preds_model1[i]]#llista amb les tres posicions
        data = Counter(list_preds)
        #hauria de fer un if, que si no hi ha una moda, que es quedi amb el primer
        hard_voting_preds[i]=data.most_common(1)  # Returns the highest occurring item
        #creo una llista amb la  posició i de cada model

    return hard_voting_preds
