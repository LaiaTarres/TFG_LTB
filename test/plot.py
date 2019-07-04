import matplotlib.pyplot as plt

filename_train = 'train_loss.txt' #train_acc.txt train_loss.txt
filename_val = 'val_loss.txt' #val_acc.txt val_loss.txt

x =list(range(1, 101)) #el numero d'epochs + 1

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
plt.savefig('loss_train_i_val_3.png') #loss_train_i_val_2 i acc_train_i_val_2

#per tenir la grafica en local:
#terminal en local: sftp ltarres@imatge.upc.edu
#contrasenya
#anem a PycharmProjects/test
#get val_loss.txt
#get train_loss.txt
#get train_acc.txt
#get val_acc.txt

