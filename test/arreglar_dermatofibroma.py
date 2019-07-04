#!/usr/bin/python
# -*- coding: utf-8 -*-
#ho he hagut de crear en local, però després ho hauré d'executar al servidor que és on estan les imatges
#en concret, les imatges estan a imatge/ltarres/work/data/ISIC-images/2018_JID_Editorial_Images
# i el programet aquest està a imatge/ltarres/PycharmProjects/test

import os, json
import shutil

path_to_json='/imatge/ltarres/work/data/ISIC-images/HAM10000/others/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

os.makedirs('/imatge/ltarres/work/data/ISIC-images/HAM10000/'+'dermatofibroma', exist_ok=True)

for i, elem in enumerate(json_files):
    with open(path_to_json+elem) as f:
        data=json.load(f)
    if data['meta']['clinical']['diagnosis'] == 'dermatofibroma':
        print('data' + data['meta']['clinical']['diagnosis'] + '==' + 'dermatofibroma')
        shutil.move(path_to_json + elem, '/imatge/ltarres/work/data/ISIC-images/HAM10000/' + 'dermatofibroma')  # mous el json
        elem2=elem.replace('.json', '.jpg')
        shutil.move(path_to_json + elem2, '/imatge/ltarres/work/data/ISIC-images/HAM10000/' + 'dermatofibroma')  # mous el jpg
