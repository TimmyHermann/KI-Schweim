from tkinter import *

BG_COLOR = "#283149"
BG_WRITE_COLOR = "#404B69"
BUTTON_COLOR = "#00818A"
TEXT_COLOR = "#DBEDF3"

FONT = "ARIAL 12"

class UserInterface:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run (self):
        self.window.mainloop()  
    
    def _setup_main_window(self):
        # WINDOW 
        # Title
        self.window.title("ExpressionRecognizer")
        # For fixed window size 
        self.window.resizable(width=False,height=False)
        # Configuration for width, height and background color
        self.window.configure(width=470, height=550, bg=BG_COLOR)


        #LAYOUT
        #headLabel
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Expression to Emoji", font=FONT, pady=10)
        # Label position (label complete window width, thats why 1) 
        head_label.place(relwidth=1)
        
        #divider
        line = Label(self.window, width=450, bg=BG_COLOR)
        line.place(relwidth=1,rely=0.07, relheight=0.012)

        #text widget (thread)
        self.text_widget = Text(self.window, width=20,height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=NORMAL)

        #scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1,relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

         # writing background
        write_bg = Label(self.window, bg=BG_WRITE_COLOR, height=80)
        write_bg.place(relwidth=1, rely=0.825)

        #message entry
        self.msg_entry = Entry(write_bg, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self._on_enter_press)

    def _on_enter_press(self,event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
    
    def _insert_message(self,msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0,END)
        msg1 = f"{msg}"
        self.text_widget.insert(END,msg1) 
        self.text_widget.configure(state=NORMAL)
        

if __name__ == "__main__":
    chat = UserInterface()
    # Class call
    chat.run()