# Install CUDA for PyTorch (YOLOv7) for Windows
You must install CUDA to use our YOLO v7 custom model. It is necessary because only if the model run with a GPU it will run smooth. This guide assumes that you have Python 3 and pip installed. <br>
To install everything please follow the following steps:<br>
1. **Install Visual Studio** <br>
First you must install Visual Studio (Yes Visual Studio NOT Visual Studio Code). It is necessary because CUDA will grumble if it is not installed. Install die latest Community Version of Visual Studio (current Version 2022). To download Visual Studio click this [link](https://visualstudio.microsoft.com/de/downloads/) or search for it manually.<br>
Follow the instructions and install just the core. The core without extra packages will be enough. If you install more it should be also ok. <br>
<br>

2. **Install CUDA 11.7** <br>
First check if your GPU runs CUDA. To check visit this [overview](https://developer.nvidia.com/cuda-gpus) from Nvidia. Search for your GPU (probably in CUDA-Enabled GeForce and TITAN Products). Example: My GeForce 2070 can run CUDA 11.7. We use CUDA Version 11.7, because it is at the moment (22.02.2023) the latest version that is supported by PyTorch. To install CUDA click this [link](https://developer.nvidia.com/cuda-11-7-0-download-archive) or search for Nvidia CUDA 11.7 manually via a search engine. On the website choose the operating system (Windows) and the Windows version you want to run it on (in my case it was WIndows 10). While installing follow the instructions and chosse the express option when installing. If it grumbles about Visual Studio it should be fine becuase Visual Studio is installed anyway. <br> 
<br>

3. **Install the right PyTorch Version** <br>
Go to the PyTorch [website](https://pytorch.org/get-started/locally/) and generate the command for installing the right PyTorch Version. You can also use the command provided in this guide. The latest pip command with stable build for Windows for CUDA 11.7 is: <br>
<br>
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
`
<br>
<br>

4. **Verify the installation** <br>
Verfy PyTorch:<br>
With `pip list` you should get the result: <br>
torch                        1.13.1+cu117 <br>
torchaudio                   0.13.1+cu117 <br>
torchvision                  0.14.1+cu117 <br>
<br>
If you do not get that output, deinstall the torch packages and install ist again, because it is possible that PyTorch was already installed with a different version. You can also make yourself a virtual environment with Conda so you can have different version at the same time, but this is not the topic of this guide.<br>
<br>
You can also run the provided test_gpu-py script to verify that PyTorch can access the GPU.



