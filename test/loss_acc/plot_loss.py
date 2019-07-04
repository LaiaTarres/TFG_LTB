import sys
import matplotlib.pyplot as plt

print('argument_list: [model,epoch,learning_rate,batch_size]',argument_list)
print('for example: [resnet18, 100, 6, 36]')
model=sys.argv[1]
epochs=int(sys.argv[2]) #integral positive
learning_rate=float('1e-'+str(sys.argv[3])) #float
batch_size=int(sys.argv[4]) #positive integral value

filename_train='train_loss_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.txt'
filename_train='val_loss_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.txt'

#filename_train = 'train_loss_resnet18_70ep_lr6_bs36_no_balanced.txt' #train_acc.txt train_loss.txt
#filename_val = 'val_loss_resnet18_70ep_lr6_bs36_no_balanced.txt' #val_acc.txt val_loss.txt

#x =list(range(1, 101)) #el numero d'epochs + 1

with open(filename_train, 'r') as textFile:
  data1 = textFile.readlines()

y = [float(s.replace('\n', '')) for s in data1]

with open(filename_val, 'r') as textFile:
  data2 = textFile.readlines()

z = [float(s.replace('\n', '')) for s in data2]


plt.plot(y, label='train')
plt.plot(z, label='val')
plt.title('loss train vs val')
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.show()
plt.savefig('loss_train_i_val_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.png') #loss_train_i_val_2 i acc_train_i_val_2

#per tenir la grafica en local:
#terminal en local: sftp ltarres@imatge.upc.edu
#contrasenya
#anem a PycharmProjects/test
#get val_loss.txt
#get train_loss.txt
#get train_acc.txt
#get val_acc.txt

