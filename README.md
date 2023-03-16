# KI-Schweim-Semester-6

## Beschreibung
Alles für das KI-Projekt im 6. Semester ist im Ordner "Sem6" zu finden.  

## Setup Guide
1. **Benötigte Versionen von CUDA und PyTorch installieren (optional, aber empfohlen)**<br>
[CUDA Installationsguide](./docs/intall_cuda_readme.md)
2. **Benötigte Bibliotheken installieren**<br>
`pip install -r requirements.txt` 
3. **Starten**
`python detect.py`
detect.py ist standardmäßig auf gpu (0) eingestellt. Wenn keine GPU vorhanden ist oder CUDA nicht funktioniert bitte auf 'cpu' in der detect.py ändern. 

## Quellen
- Base Model fürs Trainieren: [Yolo v7](https://github.com/WongKinYiu/yolov7)  
- Model dependencies (Ordner: utils, models): [Yolo v7](https://github.com/WongKinYiu/yolov7)
- Dateien/Skripte: test.py, detect.py, .gitignore, requirements.txt: [Yolo v7](https://github.com/WongKinYiu/yolov7)