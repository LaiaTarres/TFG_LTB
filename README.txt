Este repositorio contiene el código necesario para el desarrollo del proyecto:
Skin Lesion Classification with Residual Neural Network Ensemble. UPC, 2019.

Dentro de la carpeta de test están todos los ficheros. 

Para generar los diferentes modelos se ha usado:
main_my_resnet_2018.py --modelos individuales
main_my_resnet_2019.py
new_ensemble_model_output.py --ensembles
new_ensemble_model_output_4.py

Luego, para generar las gráficas se ha usado:
main_model_to_file.py -- para tener el modelo definido en un fichero
/loss_acc/plot_acc_balanced.py
/loss_acc/plot_loss_balanced.py

Finalmente, para generar la matriz de confusión:
main_confusion_matrix_cpu.py
main_confusion_matrix_cpu_2019.py
