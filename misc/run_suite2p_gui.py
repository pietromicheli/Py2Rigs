import tkinter as tk
import tkfilebrowser
from tkinter import scrolledtext
import subprocess
import os
import sys
import time
import threading
from run_suite2p import runs2p


class StdoutRedirector:

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END) 
    
    def flush(self):
        pass

def run_program(data_dirs,save_dirs,merge_bool,out):
    
    out.insert(tk.END, '\n\n||| START BATCH PROCESSING |||\n')
    merge = merge_bool.get()
    runs2p(data_dirs,save_dirs,merge)
    out.insert(tk.END, '\n||| END! |||')


def select_data_folders(dirs = []):

    data_folder_paths.extend(tkfilebrowser.askopendirnames(initialdir='Y:\\Group_folder\\recordings'))
    if data_folder_paths:
        out.insert(tk.END,'>Data folders added:\n')
        for path in data_folder_paths:
            out.insert(tk.END, '    '+path+'\n')

def select_save_folders(dirs = []):
    
    save_folder_paths.extend(tkfilebrowser.askopendirnames())
    if save_folder_paths:
        out.insert(tk.END,'>Save folders added:\n')
        for path in save_folder_paths:
            out.insert(tk.END, '    '+path+'\n')

def clear_folders():

    global data_folder_paths,save_folder_paths

    data_folder_paths = []
    save_folder_paths = []
    boolvar = False
    merge_tick.deselect()
    out.delete('1.0', tk.END)

def run_button_callback(merge_bool,out):

    if data_folder_paths:
        threading.Thread(target=run_program, args=(data_folder_paths,save_folder_paths,merge_bool,out)).start()
    else:
        print("Please select a valid Data folder before running.")

# Create the main application window
app = tk.Tk()
swidth = app.winfo_screenwidth()
sheight = app.winfo_screenheight()

app.configure(bg='black')
app.geometry("%dx%d"%(int(swidth/4),int(sheight/3)))
app.title("SUITE2P BATCH")

button_h = 30
sbar = 20
button_w = (int(swidth/4)-sbar)/5


out = tk.Text(app, wrap="none", bg='black', fg='green1')
out.place(x=0,y=button_h, width=int(swidth/4)-2*sbar, height=int(sheight/3)-3*sbar-button_h)
outVsb = tk.Scrollbar(app, orient="vertical", command=out.yview)
outHsb = tk.Scrollbar(app, orient="horizontal", command=out.xview)
out.configure(yscrollcommand=outVsb.set, xscrollcommand=outHsb.set)
outVsb.place(x=int(swidth/4)-2*sbar,y=button_h, width=sbar, height=int(sheight/3)-3*sbar-button_h)
outHsb.place(x=0,y=int(sheight/3)-3*sbar, width=int(swidth/4)-sbar, height=sbar)

sys.stdout = StdoutRedirector(out)
sys.stderr = StdoutRedirector(out)


global data_folder_paths,save_folder_paths
data_folder_paths = []
save_folder_paths = []


data_dirs = tk.Button(app, text='Data folders*', command=select_data_folders)
data_dirs.place(x=0,y=0, width=button_w, height=button_h)
data_dirs.configure(bg='purple2')

save_dirs = tk.Button(app, text='Save folders', command=select_save_folders)
save_dirs.place(x=button_w,y=0, width=button_w, height=button_h)
save_dirs.configure(bg='purple2')

global boolvar
merge_bool = tk.BooleanVar()
merge_bool.set(False)

merge_tick = tk.Checkbutton(app, text = "merge?", variable = merge_bool)
merge_tick.place(x=button_w*2,y=0, width=button_w, height=button_h)
merge_tick.configure(bg='orange2')

clear = tk.Button(app, text='CLEAR', command=clear_folders)
clear.place(x=button_w*3,y=0, width=button_w, height=button_h)
clear.configure(bg='red2')

run_button = tk.Button(app, text="RUN !", command= lambda:run_button_callback(merge_bool,out))
run_button.place(x=button_w*4,y=0, width=button_w, height=button_h)
run_button.configure(bg='green2')

# Start the Tkinter event loop
app.mainloop()