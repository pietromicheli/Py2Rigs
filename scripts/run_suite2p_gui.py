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

    data_folder_paths.extend(tkfilebrowser.askopendirnames())
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
app.title("SUITE2P BATCH")

out = tk.Text(app, wrap="none", height=20,borderwidth=0, bg='black', fg='green')
outVsb = tk.Scrollbar(app, orient="vertical", command=out.yview)
outHsb = tk.Scrollbar(app, orient="horizontal", command=out.xview)
out.configure(yscrollcommand=outVsb.set, xscrollcommand=outHsb.set)

out.grid(row=4, column=0, columnspan=5, sticky="nsew")
outVsb.grid(row=4, column=6 ,sticky="ns")
outHsb.grid(row=5, column=0, columnspan=5, sticky="ew")

sys.stdout = StdoutRedirector(out)
sys.stderr = StdoutRedirector(out)


global data_folder_paths,save_folder_paths
data_folder_paths = []
save_folder_paths = []

data_dirs = tk.Button(app, text='Data folder(s)*', command=select_data_folders)
data_dirs.grid(row=1, column=0, sticky="W")

save_dirs = tk.Button(app, text='Save folder(s)', command=select_save_folders)
save_dirs.grid(row=1, column=1,sticky="W")

global boolvar
merge_bool = tk.BooleanVar()
merge_bool.set(False)

merge_tick = tk.Checkbutton(app, text = "merge", variable = merge_bool)
merge_tick.grid(row=1, column=2,sticky="W")

clear = tk.Button(app, text='clear', command=clear_folders)
clear.grid(row=1, column=3, ipadx=8)

run_button = tk.Button(app, text="RUN !", command= lambda:run_button_callback(merge_bool,out))
run_button.grid(row=2, column=0, columnspan=5, pady=20)

# Start the Tkinter event loop
app.mainloop()