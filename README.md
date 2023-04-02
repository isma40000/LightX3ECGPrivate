This is the official source code of our paper, <br />
[LightX3ECG: A Lightweight and eXplainable Deep Learning System <br />for 3-lead Electrocardiogram Classification](https://drive.google.com/file/d/1eyFhzr0VDOi1IrJ4Cah8IuE00HiE32QG/view?usp=sharing), <br />
which is the 1st part of our project on Reduced-lead ECG Classification. Check out the 2nd part [here](https://github.com/lhkhiem28/X3ECGpp). 
# Source
## nets
## tools
### **train.py**
Para lanzar el train ahora mismo se utiliza el siguiente comando:
```
python3 tools/train.py --dataset "CPSC-2018" --num_classes 9 --multilabel
```
- dataset : Dataset que se usará para entrenar el modelo. Puede ser CPSC-2018 o Chapman. <br/>
- multilabel : Booleano que se usa para indicar si se usarán múltiples etiquetas. <br/>
- num_gpus : Número de GPUs que se usarán durante el entrenamiento. <br/>

### **Casos CPSC-2018**
Los casos guardados en CPSC-2018 son divididos por 200 y sin dividir, los que no han sido divididos son los que acaban en OG (original).

#### **Entrenamiento**
Para el entrenamiento y predicción los casos están divididos en tres subconjuntos, train, pred y val. En el train están la mayor parte de los casos, luego otra parte está en validación y finalmente los casos de la tabla de enfermedades están únicamente en pred.

#### **Aumentar cantidad de datos de Github**
git config http.postBuffer 1048576000

#### **Copiar datos**
cp -R -u -p ./scratch/tfg_luis_2223/drive/Shareddrives/TFG_INFO/Codigo/Casos/Chapman/CasosNumpy/ ./scratch/tfg_luis_2223/LightX3ECGPrivate/datasets/Casos/Chapman/CasosNumpy/