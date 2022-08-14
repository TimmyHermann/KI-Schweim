from tkinter import *

BG_COLOR = "#283149"
BG_WRITE_COLOR = "#404B69"
BG_LABEL_COLOR = "#404B69"
BUTTON_COLOR = "#00818A"
TEXT_COLOR = "#DBEDF3"

FONT = "ARIAL 12"

def _on_enter_press(event):
    msg = msg_entry.get()
    _insert_message(msg, "You")
    
def _insert_message(msg, sender):
    if not msg:
        return
    msg_entry.delete(0,END)
    msg1 = f"{msg}"
    chat_history.insert(END,msg1) 
    chat_history.configure(state=NORMAL)


root = Tk()
root.title("Macrohard Memes")
root.configure(width=900, height=900, bg=BG_COLOR)
#root.resizable(width=False,height=False)

# Head label for heading
head_label = Label(bg=BG_LABEL_COLOR, fg=TEXT_COLOR, text="Macrohard Memes Meeting", font=FONT, pady=10)
# Label position (label complete window width, thats why 1) 
head_label.place(relwidth=1)

#text widget (thread)
chat_history = Text(width=20,height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
chat_history.place()
chat_history.configure(cursor="arrow", state=NORMAL)

# writing background
write_bg = Label(bg=BG_WRITE_COLOR, height=80)
write_bg.place(relwidth=0.5, rely=0.9, relx=0.5)

#message entry
msg_entry = Entry(write_bg, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
msg_entry.focus()
msg_entry.bind("<Return>",_on_enter_press)

#send button
send_button = Button(write_bg, text="Senden", fg=TEXT_COLOR, font=FONT, width=20, bg=BG_COLOR, command=lambda: _on_enter_press(None))
send_button.place(relx=0.77,rely=0.008, relheight=0.06, relwidth=0.22)




root.mainloop()

