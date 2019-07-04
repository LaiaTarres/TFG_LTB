import glob
from sklearn.model_selection import train_test_split
import os
from shutil import copyfile

import pandas as pd

if __name__ == '__main__':
    files = glob.glob('/imatge/ltarres/work/data/ISIC2019/ISIC_2019_Training_Input/**/*.jpg', recursive=True) #directori on estan totes les imatges
    df = pd.DataFrame({'file': files})

    #print(df)

    df['class'] = df['file'].map(lambda x: x.split('/')[-2])

    files_train, files_val = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=0) #canviar lo de test size

    for file in files_train['file'].values.tolist():
        dst_path = file.replace('ISIC_2019_Training_Input', 'train') #posem la direccio on ens interessa
        os.makedirs('/'.join(dst_path.split('/')[:-1]), exist_ok=True)
        print(dst_path)
        copyfile(file, dst_path)

    for file in files_val['file'].values.tolist():
        dst_path = file.replace('ISIC_2019_Training_Input', 'val') #posem la direccio on ens interessa
        os.makedirs('/'.join(dst_path.split('/')[:-1]), exist_ok=True)
        print(dst_path)
        copyfile(file, dst_path)