# Install CUDA for PyTorch (YOLOv7) for Windows
CUDA muss istalliert werden, damit unser Yolov7 Modell flüssig läuft. In dieser Anleitung gehen wir davon aus, dass Python 3 bereits installiert ist. 
 <br>
Zum installieren bitte folgende Schritte befolgen:<br>
1. **Visual Studio installieren** <br>
Zuerst muss Visual Studio installiert werden (Ja Visual Studio NICHT Visual Studio Code). Dies ist nötig, da sonst CUDA bei der Installation ein Problem meldet. Dies ist in Windows nötig, wenn ein paar Dependencies. Einfach die neueste Community Version installieren (aktuellste: 2022). Um Visual Studio runterzuladen hier auf den [Link](https://visualstudio.microsoft.com/de/downloads/) clicken, sonst manuell ruinterladen. Bei der Installation ist nur das Core-Package wichtig. <br>
<br>

2. **CUDA 11.7 installieren** <br>
Zuerst überprüfen ob die eigene GPU für CUDA geeinget ist. Nvidia hat hierzu eine [Übersicht](https://developer.nvidia.com/cuda-gpus). Wir benutzen CUDA 11.7, weil das die neueste Version ist, welche von PyTorch unterstützt wird (zum Zeitpunkte 22.02.2023). Einfach hier auf den [Link](https://developer.nvidia.com/cuda-11-7-0-download-archive) clicken und die Windows Version auswählen und herunterladen. Bei der Installation die Express Option wählen und installieren. Wenn CUDA meckert, das was mit Visual Studio nicht stimmen würde, müsste dies so passen, da Visual Studio bereits installiert ist. <br>
<br>

3. **Die richtige PyTorch Version installieren** <br>
Um die richtigen PyTorch Versionen zu installieren diesen Befehl ausführen. Wenn bereits PyTorch installiert ist, bitte diese vorher entfernen oder eine virtuelle Umgebung mit Conda erstellen. 
<br>
<br>
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
`
<br>
<br>

4. **Verifizierung der Installation** <br>
Das sollte das Ergebnis sein:<br>
Hierfür `pip list`: <br>
torch                        1.13.1+cu117 <br>
torchvision                  0.14.1+cu117 <br>
<br>
<br>
Das im ./docs Ordner bereitgestellte test_gpu.py Skript kann zur Verifizierung verwendet werden, ob die GPU erkannt wird. 


