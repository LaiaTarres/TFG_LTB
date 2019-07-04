import csv, os, torch
import shutil

path = '/imatge/ltarres/work/data/ISIC2019/ISIC_2019_Training_Input/'
file = '/imatge/ltarres/work/data/ISIC2019/ISIC_2019_Training_GroundTruth.csv'

#image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK
os.makedirs(path+'melanoma', exist_ok=True)
os.makedirs(path+'nevus', exist_ok=True)
os.makedirs(path+'basal_cell_carcinoma', exist_ok=True)
os.makedirs(path+'actinic_keratosis', exist_ok=True)
os.makedirs(path+'benign_keratosis', exist_ok=True)
os.makedirs(path+'dermatofibroma', exist_ok=True)
os.makedirs(path+'vascular_lesion', exist_ok=True)
os.makedirs(path+'squamous_cell_carcinoma', exist_ok=True)
os.makedirs(path+'none_of_the_others', exist_ok=True)



name = [12]

with open(file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0

    for row in csv_reader:
        count = 0
        for elem in row:
            count += 1
            name.insert(count, elem)
            #print('elem' + elem, '\n')
            #print('count' + str(count), '\n')
            #print('name' + name[1], '\n')
            if count == 2 and elem == str(1.0):  # MEL
                shutil.move(path + name[1] + '.jpg', path + 'melanoma')
                print('mel', '\n')
            elif count == 3 and elem == str(1.0):  # NV
                shutil.move(path + name[1] + '.jpg', path + 'nevus')
                print('nevus', '\n')
            elif count == 4 and elem == str(1.0): #BCC
               shutil.move(path + name[1] + '.jpg' , path + 'basal_cell_carcinoma')
               print('bcc', '\n')
            elif count == 5 and elem == str(1.0):  #AK
                shutil.move(path + name[1] + '.jpg' , path + 'actinic_keratosis')
            elif count == 6 and elem == str(1.0):  #BKL
                shutil.move(path + name[1] + '.jpg' , path + 'benign_keratosis')
                print('benign_keratosis', '\n')
            elif count == 7 and elem == str(1.0):  #DF
                shutil.move(path + name[1] + '.jpg', path + 'dermatofibroma')
                print('DF', '\n')
            elif count == 8 and elem == str(1.0):  # VASC
                shutil.move(path + name[1] + '.jpg', path + 'vascular_lesion')
                print('VASC', '\n')
            elif count == 9 and elem == str(1.0):  #SCC
                shutil.move(path + name[1] + '.jpg' , path + 'squamous_cell_carcinoma')
                print('SCC', '\n')
            elif count == 10 and elem == str(1.0): #UNK
                shutil.move(path + name[1] + '.jpg' , path + 'none_of_the_others')
                print('UNK', '\n')

#per trobar el directori .csv