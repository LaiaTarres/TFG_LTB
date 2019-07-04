from __future__ import print_function, division

labels = [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0] # 0: tp:13 tn:1 fp:1 fn:1 , 1: tp:1 tn:13 fp:1 fn:1 , resta: tp:0 tn:16 fp:0 fn:0
preds  = [0,1,0,1,0,2,0,0,0,0,0,0,0,0,0,0]

true_positive=[0,0,0,0,0,0,0]
true_negative=[0,0,0,0,0,0,0]
false_positive=[0,0,0,0,0,0,0]
false_negative=[0,0,0,0,0,0,0]

for j in range(0,len(labels)): #el vector labels va de 0 a 15
    for i in range(0,len(true_positive)): #volem que tingui les 7 classes, el vector va de 0 a 6
        if labels[j] == i:
            if preds[j]==i:
                true_positive[i]=true_positive[i] + 1
            else :
                false_negative[i] = false_negative[i] + 1
        else :
            if preds[j]==i:
                false_positive[i]=false_positive[i] + 1
            else:
                true_negative[i] = true_negative[i] + 1

for i in range(0,len(true_positive)):
    print('true_positive['+str(i)+']:'+str(true_positive[i]))
    print('true_negative['+str(i)+']:'+str(true_negative[i]))
    print('false_negative['+str(i)+']:'+str(false_negative[i]))
    print('false_positive['+str(i)+']:'+str(false_positive[i]))

print('true_positive:'+str(true_positive))
print('true_negative:'+str(true_negative))
print('false_negative:'+str(false_negative))
print('false_positive:'+str(false_positive))

TP=sum(true_positive)/len(true_positive)
TN=sum(true_negative)/len(true_negative)
FN=sum(false_negative)/len(false_negative)
FP=sum(false_positive)/len(false_positive)
print(TP)
print(TN)
print(FN)
print(FP)
epoch_acc=(TP/(TP+FN)+TN/(TN+FP))/2
print(epoch_acc)