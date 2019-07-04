import matplotlib.pyplot as plt
import sys

#argument_list=str(sys.argv)
#print('argument_list: [model,epoch,learning_rate,adapt_lr,batch_size]',argument_list)
#print('for example: [resnet18, 100, 6, na/a, 36]')
#model=sys.argv[1]
#epochs=int(sys.argv[2]) #integral positive
#learning_rate=float('1e-'+str(sys.argv[3])) #float
#a_learning_rate=sys.argv[4]
#batch_size=int(sys.argv[5]) #positive integral value

#filename_train='train_acc_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+a_learning_rate+'_bs'+str(batch_size)+'_balanced.txt'
#filename_val='val_acc_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+a_learning_rate+'_bs'+str(batch_size)+'_balanced.txt'

#filename_train = 'train_acc_resnet18_70ep_lr6_bs36_balanced.txt' #train_acc.txt train_loss.txt
#filename_val = 'val_acc_resnet18_70ep_lr6_bs36_balanced.txt' #val_acc.txt val_loss.txt

model_param=sys.argv[1]
print('El model que has passat per parametre es: ',model_param)

print('srun python plot_acc_balanced.py my_resnet_101_2018')
filename_train='train_acc_'+model_param+'.txt'
filename_val='val_acc_'+model_param+'.txt'
#x =list(range(1, 101)) #el numero d'epochs + 1

with open(filename_train, 'r') as textFile:
  data1 = textFile.readlines()

y = [float(s.replace('\n', '')) for s in data1]

with open(filename_val, 'r') as textFile:
  data2 = textFile.readlines()

z = [float(s.replace('\n', '')) for s in data2]


plt.plot(y, label='train')
plt.plot(z, label='val')
plt.title('acc balanced train vs val')
plt.xlabel('epoch')
plt.ylabel('acc')
#plt.show()
plt.savefig('acc_train_i_val_'+model_param)

#per tenir la grafica en local:
#terminal en local: sftp ltarres@imatge.upc.edu
#contrasenya
#anem a PycharmProjects/test
#get val_loss.txt
#get train_loss.txt
#get train_acc.txt
#get val_acc.txt

