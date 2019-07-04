import sys

model_param=sys.argv[1]
print('El model que has passat per parametre es: ',model_param)

filename_train='train_acc_'+model_param+'.txt'
filename_val='val_acc_'+model_param+'.txt'

with open(filename_train, 'r') as textFile:
  data1 = textFile.readlines()

y = [float(s.replace('\n', '')) for s in data1]
max_acc_train=max(y)

with open(filename_val, 'r') as textFile:
  data2 = textFile.readlines()

z = [float(s.replace('\n', '')) for s in data2]
max_acc_val=max(z)

print('max_acc_train:',max_acc_train)
print('max_acc_val',max_acc_val)