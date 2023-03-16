# This is a modified Version of the detect.py provided by the Yolov7 authors. Link to the repo: https://github.com/WongKinYiu/yolov7
#
# The UI is inspired by this tutorial: https://www.youtube.com/watch?v=sopNW98CRag&t=1747s
# The implementation of the video stream is inspired by this tutorial: https://www.youtube.com/watch?v=cONDuZeYdzc

# Imports for Yolov7
import argparse
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Imports for the UI
from tkinter import *

# Libraries for the camera
import numpy as np
from PIL import Image, ImageTk
import cv2

# Emojis
import emoji

# Threading
import threading

# Colors
BG_COLOR = "#283149"
BG_WRITE_COLOR = "#404B69"
BG_LABEL_COLOR = "#404B69"
BUTTON_COLOR = "#00818A"
TEXT_COLOR = "#DBEDF3"

# Fonts
FONT = "ARIAL 12"


class detectP:

    # Arguments fpr the model
    # source "0" is webcam, device either 'cpu' or '0' for gpu
    source, weights, imgs, trace, device, cthres, ithres, augment, save_conf, project, name, save_txt = '0', 'all_gestures.pt', 640, True, str(
        'cpu'), 0.60, 0.65, True, False, 'runs/detect', 'exp', False
    webcam = source.isnumeric()

    # Directories
    save_dir = Path(increment_path(Path(project) / name))  # increment run
    (save_dir / 'labels' if True else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgs, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgs)

    # half is only relevant when a GPU is used
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # UI
    # Vars
    img_height = 105
    img_width = 186
    predi = np.array([0, 0])
    l = 0
    cnt = 1
    blocked = False

    # Init of the tkinter for the GUI
    def __init__(self):
        self.window = Tk()
        threading.Thread(target=self._setup_window()).start()

    # UI must run till it is closed
    def run(self):
        self.window.mainloop()

    # This is the function with all the UI components
    def _setup_window(self):

        # Basid settings for the window
        self.window.title("Macrohard Memes v2")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=1000, height=550, bg=BG_COLOR)

        # Head label
        head_Label = Label(self.window, bg=BG_LABEL_COLOR, fg=TEXT_COLOR, text="Macrohard Memes Meeting", font=FONT,
                           pady=10)
        # relwidth 1 = full width of the window, because window is parent element
        head_Label.place(relwidth=1)

        # Chat history
        # width = how many characters in on line, height= number of lines per message
        self.chat_history = Text(self.window, width=10, height=2, bg=BG_WRITE_COLOR, fg=TEXT_COLOR, font=FONT, padx=5,
                                 pady=5)
        self.chat_history.place(relwidth=0.4, relheight=0.8, rely=0.08, relx=0.58)
        self.chat_history.configure(cursor="arrow", state=DISABLED)

        # Message entry
        self.msg_entry = Entry(self.window, bg=BG_WRITE_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.3, relheight=0.08, rely=0.9, relx=0.58)
        self.msg_entry.focus()

        # Send button
        send_button = Button(self.window, text="Senden", fg=TEXT_COLOR, font=FONT, width=20, bg=BG_COLOR)
        send_button.place(relx=0.88, rely=0.9, relheight=0.08, relwidth=0.1)

        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)

        # Camera
        self.cam = Label(self.window, bg=BG_COLOR)
        self.cam.place(relwidth=0.525, relheight=0.9, rely=0.08, relx=0.03)

        # Threading for the camera
        self.detect()

    # Gesture detection
    def detect(self):

        self.dataset.__iter__()
        path, img, im0s, vid_cap = self.dataset.__next__()

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.cthres, self.ithres, agnostic=True)
        t3 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if True else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            im0 = Image.fromarray(im0)
            im0 = ImageTk.PhotoImage(im0)
            self.cam.configure(image=im0)
            self.cam.image = im0
            cv2.waitKey(1)  # 1 millisecond
            threading.Thread(target=self.detect).start()

if __name__ == '__main__':
    MH_Memes = detectP()
    MH_Memes.run()

