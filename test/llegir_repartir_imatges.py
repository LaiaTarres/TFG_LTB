#!/usr/bin/python
# -*- coding: utf-8 -*-
#ho he hagut de crear en local, però després ho hauré d'executar al servidor que és on estan les imatges
#en concret, les imatges estan a imatge/ltarres/work/data/ISIC-images/2018_JID_Editorial_Images
# i el programet aquest està a imatge/ltarres/PycharmProjects/test

import os, json
import pandas as pd
import shutil

path_to_json='/imatge/ltarres/work/data/ISIC-images/HAM10000/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#print(json_files)  # for me this prints ['foo.json']

#creem les carpetes
os.makedirs(path_to_json+'melanoma', exist_ok=True)
os.makedirs(path_to_json+'squamous_cell_carcinoma', exist_ok=True)
os.makedirs(path_to_json+'basal_cell_carcinoma', exist_ok=True)
os.makedirs(path_to_json+'nevus', exist_ok=True)
os.makedirs(path_to_json+'pigmented_cell_carcinoma', exist_ok=True)
os.makedirs(path_to_json+'pigmented_benign_keratosis', exist_ok=True)
os.makedirs(path_to_json+'actinic_keratosis', exist_ok=True)
os.makedirs(path_to_json+'vascular_lesion', exist_ok=True)
os.makedirs(path_to_json+'dermatofibroma', exist_ok=True)


#per saber l'estructura, a part he fet: python -m json.tool ISIC_0036065.json per un altre terminal
for i, elem in enumerate(json_files):
    with open(path_to_json+elem) as f:
        data=json.load(f)
    #print(data['meta']['clinical']['diagnosis']) #aqui dins hi ha el diagnosi
    if data['meta']['clinical']['diagnosis']=='melanoma':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'melanoma')
        shutil.move(path_to_json+elem,path_to_json+'melanoma')#mous el json
        elem2=elem.replace('.json','.jpg')
        shutil.move(path_to_json+elem2,path_to_json+'melanoma') #mous el jpg
    elif data['meta']['clinical']['diagnosis']=='squamous cell carcinoma':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'squamous_cell_carcinoma')
        shutil.move(path_to_json + elem, path_to_json + 'squamous_cell_carcinoma')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'squamous_cell_carcinoma')  # mous el jpg
    elif data['meta']['clinical']['diagnosis']=='basal cell carcinoma':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'basal_cell_carcinoma')
        shutil.move(path_to_json + elem, path_to_json + 'basal_cell_carcinoma')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'basal_cell_carcinoma')  # mous el jpg
    elif data['meta']['clinical']['diagnosis'] == 'nevus':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'nevus')
        shutil.move(path_to_json + elem, path_to_json + 'nevus')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'nevus')  # mous el jpg
    elif data['meta']['clinical']['diagnosis'] == 'pigmented cell carcinoma':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'pigmented_cell_carcinoma')
        shutil.move(path_to_json + elem, path_to_json + 'pigmented_cell_carcinoma')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'pigmented_cell_carcinoma')  # mous el jpg
    elif data['meta']['clinical']['diagnosis'] == 'pigmented benign keratosis':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'pigmented_benign_keratosis')
        shutil.move(path_to_json + elem, path_to_json + 'pigmented_benign_keratosis')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'pigmented_benign_keratosis')  # mous el jpg
    elif data['meta']['clinical']['diagnosis'] == 'actinic keratosis':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'actinic_keratosis')
        shutil.move(path_to_json + elem, path_to_json + 'actinic_keratosis')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'actinic_keratosis')  # mous el jpg
    elif data['meta']['clinical']['diagnosis'] == 'vascular lesion':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'vascular_lesion')
        shutil.move(path_to_json + elem, path_to_json + 'vascular_lesion')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'vascular_lesion')  # mous el jpg
    elif data['meta']['clinical']['diagnosis'] == 'dermatofibroma':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'dermatofibroma')
        shutil.move(path_to_json + elem, path_to_json + 'dermatofibroma')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, path_to_json + 'dermatofibroma')  # mous el jpg

#to be continued...
#escollim unes quantes imatges de cada classe (només estem agafant la carpeta 2018_JID_Editorial_Images):
#melanoma: ISIC_0036065.json
#squamous cell carcinoma: ISIC_0024304.json, ISIC_0024305.json, ISIC_0024294.json, ISIC_0024282.json
#basal cell carcinoma: ISIC_0024299.json, ISIC_0024280.json, ISIC_0024230.json
#nevus
#pigmented cell carcinoma
#pigmented benign keratosis
#actinic keratosis
#vascular lesion
