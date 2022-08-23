# Here a object oriented approach will be used because we need the 
# tkinter components and also functions for these components
#
# The UI is inspired by this tutorial: https://www.youtube.com/watch?v=sopNW98CRag&t=1747s
# The implementation of the video stream is inspired by this tutorial: https://www.youtube.com/watch?v=cONDuZeYdzc


# Import everything from tkinter
from tkinter import *

# Libraries for the camera
import numpy as np
from PIL import Image, ImageTk
import cv2

# Library for the AI
import tensorflow as tf

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

# load Model
model = tf.keras.models.load_model('../models/modelTUS')

class MeetingUi:
    # Vars
    img_height = 105
    img_width = 186
    predi = np.array([0, 0])
    l = 0
    cnt = 1
    blocked = False

    # Initiating tkinter
    def __init__(self):
        self.window = Tk()
        threading.Thread(target=self._setup_window()).start()
        
    # UI must run till it is closed 
    def run (self):
        self.window.mainloop()

    # This is the function with all the UI components
    def _setup_window(self):

        # Basid settings for the window
        self.window.title("Macrohard Memes")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=1000, height=550, bg=BG_COLOR)

        # Head label
        head_Label = Label(self.window, bg=BG_LABEL_COLOR, fg=TEXT_COLOR, text="Macrohard Memes Meeting", font=FONT, pady=10)
        # relwidth 1 = full width of the window, because window is parent element
        head_Label.place(relwidth=1)    

        # Chat history
        # width = how many characters in on line, height= number of lines per message 
        self.chat_history = Text(self.window, width=10, height=2, bg=BG_WRITE_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.chat_history.place(relwidth=0.4, relheight=0.8, rely=0.08, relx=0.58)
        self.chat_history.configure(cursor="arrow", state=DISABLED)

        # Message entry
        self.msg_entry = Entry(self.window, bg=BG_WRITE_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.3, relheight=0.08, rely=0.9, relx=0.58)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter)

        # Send button
        send_button = Button(self.window, text="Senden", fg=TEXT_COLOR, font=FONT, width=20, bg=BG_COLOR)
        send_button.place(relx=0.88,rely=0.9, relheight=0.08, relwidth=0.1)

        # Camera
        self.cam = Label(self.window, bg=BG_COLOR)
        self.cam.place(relwidth=0.525, relheight=0.9, rely=0.08, relx=0.03)

        # Capture from camera
        self.cap = cv2.VideoCapture(0)
        # Threading for the camera
        threading.Thread(target=self.video_stream()).start()

    # Function for video streaming
    def video_stream(self):

        _,frame= self.cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        self.frame_check_KI(frame)
        frame = ImageTk.PhotoImage(frame)
        self.cam.configure(image=frame)
        self.cam.image=frame
        self.cam.after(1,self.video_stream)

    def frame_check_KI(self,frame):
        im = frame.resize((self.img_width, self.img_height))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, batch_size=None, verbose=0)
        print(prediction)

        #ToDo ggf. mehr als 50%
        if np.argmax(prediction, axis=1) == 0:
            self.blocked = False

        if self.l == 10:
            if not self.blocked:
                self.predi = np.append(self.predi, np.argmax(prediction, axis=1))
                if sum(self.predi[-3:]) == 3:
                    print("Thumbs Up Nr: " + str(self.cnt))
                    self.cnt = self.cnt + 1
                    self.post_Thumbs_Up()
                    self.blocked = True
                self.predi = self.predi[-6:]
            self.l = 0
        self.l = self.l + 1

        # F-Score

    def _on_enter(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg)

    def _insert_message(self, msg):
        if not msg:
            return
        self.msg_entry.delete(0,END)
        msg1 = f"{msg}"
        # Enable writing for the chat history 
        self.chat_history.configure(state=NORMAL)
        self.chat_history.insert(END,msg1+'\n') 
        # Disable writing again for the chat history 
        self.chat_history.configure(state=DISABLED)

    def post_Thumbs_Up(self):
        self.msg_entry.insert(0, emoji.emojize(':thumbs_up:')+'\n')
        self._on_enter(None)

# Main function 
if __name__== "__main__":
    MH_Memes = MeetingUi()
    MH_Memes.run()

