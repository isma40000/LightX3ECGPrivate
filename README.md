## **Referencias**
### **Repositorios**
Repositorio del que se ha utilizado código:

- [LightX3ECG: A Lightweight and eXplainable Deep Learning System <br />for 3-lead Electrocardiogram Classification - Repositorio 1](https://github.com/lhkhiem28/LightX3ECG)<br />

Repositorios asociados con los datos de TNMG:

- [ECG Preprocessing - Repositorio 2](https://github.com/antonior92/ecg-preprocessing/tree/main)<br />
- [Automatic diagnosis of the 12-lead ECG using a deep neural network - Repositorio 3](https://github.com/lhkhiem28/LightX3ECG)<br />

### **Datos**
Aquí están los distintos datasets utilizados:
- [CPSC-2018](http://2018.icbeb.org/Challenge.html) $\rightarrow$ Los datos están en mV. El dataset se encuentra dividido en [Parte 1](http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip), [Parte 2](http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip) y [Parte 3](http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip).<br />
- [Chapman](https://figshare.com/collections/ChapmanECG/4560497/2) $\rightarrow$ Elegir ECGData.zip, los datos están en ${\mu}V$.<br />
- [TNMG](https://github.com/antonior92/ecg-preprocessing/tree/main) $\rightarrow$ En el repositorio del enlace puedes encontrar los enlaces a los datos de [prueba](https://zenodo.org/record/3765780) y una parte de los de [entrenamiento](https://zenodo.org/record/4916206). Estos datos han sido preprocesados para que se ajusten a las necesidades del repositorio 3.<br />

## **Entrenamiento**
### **Parámetros de entrenamiento y predicción**
Los parámetros de entrenamiento pueden ser configurados en el fichero train.py encontrado dentro de la carpeta tools, igualmente para configurar los parámetros de la predicción modificar el fichero predict.py .Algunos de los parámetros más interesantes son los siguientes:

Config:
- ecg_leads $\rightarrow$ Las derivaciones que deseas utilizar para entrenar.<br/>
- ecg_length $\rightarrow$ La longitud de los exámenes de ECG.<br/>

train_loaders:
- df_path $\rightarrow$ Ruta donde esté situado el CSV con la lista de casos de entrenamiento/validación/predicción y sus respectivas etiquetas (train.csv, val.csv, pred.csv). El de predicción no necesita tener las etiquetas reales, las columnas de las etiquetas pueden estar a ceros.<br/>
- data_path $\rightarrow$ Ruta donde estén situados los casos con los que se desea entrenar/validar/predecir en formato numpy (CasosNumpy(CPSC-2018)/ECGDataRawNpy-1000(Chapman))<br/>

train_fn():
- num_epochs $\rightarrow$ La cantidad de épocas durante las cuales quieres entrenar.<br/>
### **Entrenamiento**
Para poder entrenar hay que estar situdado en &lt;CarpetaEstéClonadoElRepositorio>/LightX3ECGPrivate/source , se puede utilizar los siguientes comandos, cada uno correspondiente a un dataset:
```
python3 tools/train.py --dataset "CPSC-2018" --num_classes 9 --multilabel --num_gpus 4
python3 tools/train.py --dataset "Chapman" --num_classes 4 --num_gpus 4
python3 tools/train.py --dataset "Antonio" --num_classes 6 --multilabel --num_gpus 4
```
- dataset : Dataset que se usará para entrenar el modelo. Puede ser CPSC-2018 o Chapman. <br/>
- num_classes : Número de clases distintas que pueden estar presentes en los datos. <br/>
- multilabel : Booleano que se usa para indicar si se usarán múltiples etiquetas. <br/>
- num_gpus : Número de GPUs que se usará durante el entrenamiento. Está limitado por las características de la máquina utilizada.<br/>
### **Predicción**
```
python3 tools/predict.py --dataset "CPSC-2018" --num_classes 9 --multilabel --num_gpus 4
python3 tools/predict.py --dataset "Chapman" --num_classes 4 --num_gpus 4
python3 tools/predict.py --dataset "Antonio" --num_classes 6 --multilabel --num_gpus 4
```
## Anotaciones
Los casos de TNMG no están subidos al repositorio debido a que ocupan demasiado. Se puede descargar de [aquí](https://zenodo.org/record/4916206), una vez descargados los archivos hdf5 deberían estar colocados en una misma carpeta para poder ejecutar la función que los transforma a numpy una única vez para todos los casos, H5toCSV() definida en PrediccionesLightX3ECG.ipynb .