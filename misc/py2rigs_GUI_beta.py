import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pathlib import Path
import threading
import numpy as np
from Py2Rigs import rec, sync, loader, batch
from Py2Rigs.plot import *

INPUT_LIST = ['data_paths','sync_paths','stimdict_paths','behavior_paths']

TEST_PATHS = [r"Y:\Pietro\2Precordings\PM_230224_OBO041195\rec_000_001\suite2p\plane0",
              r"Y:\Pietro\2Precordings\PM_230224_OBO041195\rec_000_001\rec_000_001.mat",
              r"Y:\Pietro\2Precordings\PM_230224_OBO041195\rec_000_001\stim_dict_PM_230224_OBO041195_001.json",
              r"Y:\Pietro\2Precordings\PM_230224_OBO041195\rec_000_001\contra_pupil_001.npy" ]

# TEST_PATHS = [r"Y:\Pietro\2Precordings\PM_050424_OBO042444\rec_000_000\suite2p\plane0",
#               r"Y:\Pietro\2Precordings\PM_050424_OBO042444\rec_000_000\rec_000_000.mat",
#               r"Y:\Pietro\2Precordings\PM_050424_OBO042444\rec_000_000\stim_dict_PM_050424_OBO042444_000.json",
#                '' ]

class StdoutRedirector:

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END) 
    
    def flush(self):
        pass

class InputFrame:
    
    def __init__(self, master, run_command):

        ttk.Style().configure('TFrame', background='black')
        ttk.Style().configure("BW.TLabel", foreground="white", background="black", font=("Arial", 10))
        ttk.Style().configure("BW.TButton", foreground="black", borderwidth=1, background="red", font=("Arial", 10))
        ttk.Style().configure("S.TLabel", foreground="white", background="black", font=("Arial", 14))
        ttk.Style().configure("S.TButton", foreground="black", borderwidth=1, background="red", font=("Arial", 14))

        self.master = master
        self.run_command = run_command

        # input section
        self.input_paths = {ptype:tk.StringVar() for ptype in INPUT_LIST}
        self.input_lists = {ptype:[] for ptype in INPUT_LIST}

        self.input_frame = ttk.Frame(self.master, style='TFrame')
        label = ttk.Label(self.input_frame, text="INPUTS", style='S.TLabel')
        label.grid(row=0, column=0, padx=0,sticky='nw')
        
        for i,(path_name, path_label) in enumerate(zip(INPUT_LIST,["* Data Path:", 
                                                                   "* Sync File (.mat):", 
                                                                   "* Event Dictionary (.json):", 
                                                                   "  Behavior (.npy):"])):
            
            label = ttk.Label(self.input_frame, text=path_label, style='BW.TLabel')
            label.grid(row=i+1, column=0, padx=0, sticky="nsew")
            entry = ttk.Entry(self.input_frame, textvariable=self.input_paths[path_name])
            entry.grid(row=i+1, column=1, pady=5, padx=5, sticky='nsew')
        
        b1 = ttk.Button(self.input_frame, text="Add Paths", style='BW.TButton', command=lambda path=path_name: self.add_paths())
        b1.grid(row=2, column=2, padx=3, pady=5, sticky='nsew')
        b2 = ttk.Button(self.input_frame, text="CLEAR", style='BW.TButton', command=self.clear)
        b2.grid(row=3, column=2, pady=5, padx=3, sticky='nsew')
        b3 = ttk.Button(self.input_frame, text="RUN", style='BW.TButton', command=self.run_callback)
        b3.grid(row=4, column=2, columnspan=2, padx=3, pady=5, sticky='nsew')
        
        # Standard Output Window
        self.output_frame = ttk.Frame(self.master, style='TFrame')
        self.output_frame.grid(row=0, column=1, columnspan=1, sticky='nsew')

        self.output_text = tk.Text(self.output_frame, wrap="word", bg='black', fg='white', bd=0, width=60, height=10)
        self.output_text.grid(row=0, column=0)
        
        self.output_text.bind("<MouseWheel>", self.scroll_text)

        sys.stdout = StdoutRedirector(self.output_text)
        sys.stderr = StdoutRedirector(self.output_text)
    
    def clear(self):

        for ptype in self.input_paths.values(): ptype.set('')  
        self.input_lists = {ptype:[] for ptype in INPUT_LIST}
        self.output_text.delete('1.0', tk.END)

    def add_paths(self):

        for i,path_name in enumerate(self.input_paths):

            path_var = self.input_paths[path_name]
            # path = TEST_PATHS[i]
            path = path_var.get()

            if path=='':
                if path_name != 'behavior_paths'and path_name != 'data_paths':
                    print(f'\nERROR: Please provide all the required paths before adding!')
                    self.clear()
                else:
                    path = None

            self.input_lists[path_name].append(path)
            path_var.set("")
            self.output_text.insert(tk.END, f"\nAdded '{path}' to {path_name} list.\n")
            self.output_text.see(tk.END)

    def run_callback(self):

        threading.Thread(target=self.run_command, kwargs={'input':self.input_lists}).start()

    def scroll_text(self, event):
        # Scroll the text widget programmatically
        if event.delta > 0:
            # Scroll up
            self.output_text.yview_scroll(-1, "units")
        else:
            # Scroll down
            self.output_text.yview_scroll(1, "units")

class PlotSection:

    def __init__(self, master, figsize):

        ttk.Style().configure('TFrame', background='black')
        ttk.Style().configure("BW.TLabel", foreground="white", background="black", font=("Arial", 10))
        ttk.Style().configure("BW.TButton", foreground="black", borderwidth=1, background="red", font=("Arial", 10))
        ttk.Style().configure("S.TLabel", foreground="white", background="black", font=("Arial", 14))
        ttk.Style().configure("S.TButton", foreground="black", borderwidth=1, background="red", font=("Arial", 14))

        self.master = master
        self.cspan = 1
        self.figsize = figsize

        self.plotting_frame = ttk.Frame(self.master, style='TFrame')
        self.plotting_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = self.add_canvas(self.figsize)
        self.canvas.get_tk_widget().grid(row=1,column=0,columnspan=self.cspan)
        self.canvas.draw()

        self.tb_fr = ttk.Frame(self.master, style='TFrame')
        self.tb_fr.grid(row=2, columnspan=self.cspan, sticky='nsew')
        self.tb = NavigationToolbar2Tk(self.canvas, self.tb_fr)

    def add_label(self, text, style=None):

        l = ttk.Label(self.plotting_frame, text=text, style=style)
        self.cspan += 1
        return l
    
    def add_entry(self, style=None):

        e = ttk.Entry(self.plotting_frame, style=style)
        self.cspan += 1
        return e
    
    def add_button(self, text, command, style=None):

        b = ttk.Button(self.plotting_frame,text=text, style=style, command=command)
        self.cspan += 1
        return b
    
    def add_checkbutton(self, text, var, style=None):
        
        cb = ttk.Checkbutton(self.plotting_frame, text=text, variable=var, 
                             onvalue=1, offvalue=0, style=style)
        self.cspan += 1
        return cb

    def add_menu(self, var, args, style=None):

        m = ttk.OptionMenu(self.plotting_frame, var, style=style, *args)
        self.cspan += 1
        return m
    
    def add_canvas(self, figsize):

        fig,ax = plt.subplots(1,1, figsize=figsize)
        ax.figure.set_facecolor('black')
        c = FigureCanvasTkAgg(fig, master=self.plotting_frame)
        return c
    
    def add_toolbar(self, canvas, row, columnspan=1):
        
        tf = ttk.Frame(self.plotting_frame)
        tf.grid(row=row, sticky='nsew', columnspan=columnspan)
        tb = NavigationToolbar2Tk(canvas, tf)
        tb.update()
        return tf, tb
    
    def update(self, fig=None):

        if fig == None:
            fig = plt.Figure(figsize=self.figsize)
        
        self.canvas.figure = fig
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=self.cspan)
        self.canvas.draw()

        if hasattr(self, 'tb_fr'):
            self.tb_fr.destroy()

        self.tb_fr = ttk.Frame(self.plotting_frame, style='TFrame')
        self.tb_fr.grid(row=2, columnspan=self.cspan, sticky='nsew')
        self.tb = NavigationToolbar2Tk(self.canvas, self.tb_fr)
        self.tb.update()

class GUI:

    def __init__(self, root):

        self.root = root
        self.f1_w = 4
        self.f1_h = 3
        self.f2_w = 15
        self.f2_h = 5
        self.f3_w = 15
        self.f3_h = 4

        self.width = 1520
        self.height = 1420
        # root.geometry("900X1200")

        master = ttk.Frame(root)
        master.grid(row=0, column=0, sticky='nsew')
        self.master = master
        self.record = None

        # TOP SECTION #
        self.first_section = InputFrame(master, self.run)
        self.first_section.input_frame.grid(row=0, column=0, sticky='nsew')
        self.first_section.output_frame.grid(row=1, column=0, columnspan=1, sticky='nsew')

        ###

        # FRST PLOT #
        self.plot1 = PlotSection(master, figsize=(self.f1_w,self.f1_h))
        self.plot1.plotting_frame.grid(row=0, column=1, rowspan=2, pady=10, padx=10, sticky='nsew')
        
        self.pops_embed = tk.StringVar()
        self.pops_k = tk.StringVar()
        self.pops_clust = tk.StringVar()

        b = self.plot1.add_button('GET POPULATIONS', self.plot_pops, "BW.TButton",)
        b.grid(row=0,column=0)

        l = self.plot1.add_label('embeding', "BW.TLabel")
        l = l.grid(row=0,column=1,sticky='w') 
        m = self.plot1.add_menu(self.pops_embed, ["pca", "pca", "umap", "tsne"])
        m.grid(row=0,column=2,sticky='w')

        l = self.plot1.add_label('n_clusers', "BW.TLabel",)
        l.grid(row=0,column=3,sticky='w')
        m = self.plot1.add_menu(self.pops_k, ["1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        m.grid(row=0,column=4,sticky='w')

        l = self.plot1.add_label('clustering', "BW.TLabel",)
        l.grid(row=0,column=5,sticky='w')
        m = self.plot1.add_menu(self.pops_clust, ["kmeans", "kmeans", "gmm"])
        m.grid(row=0,column=6,sticky='w')

        self.plot1.update()

        ###

        # SECOND PLOT #
        self.plot2 = PlotSection(master, figsize=(self.f1_w,self.f1_h))
        self.plot2.plotting_frame.grid(row=0, column=2, rowspan=2, pady=10, padx=10, sticky='nsew')
        
        self.fov_pops = tk.StringVar()

        b = self.plot2.add_button('PLOT FOV', self.plot_fov, "BW.TButton",)
        b.grid(row=0,column=0)

        l = self.plot2.add_label('populations', "BW.TLabel")
        l = l.grid(row=0,column=1,sticky='w')
        self.fov_pops_m = self.plot2.add_menu(self.fov_pops, ['None', 'None'])
        self.fov_pops_m.grid(row=0,column=2,sticky='w')

        l = self.plot2.add_label('brighness (int)', "BW.TLabel")
        l.grid(row=0,column=3,sticky='w')
        self.fov_bright_f = self.plot2.add_entry()
        self.fov_bright_f.insert(0, 'None')
        self.fov_bright_f.grid(row=0,column=4,sticky='w')

        self.plot2.update()

        ###

        # BIG PLOT 1#
        self.plot3 = PlotSection(master, figsize=(self.f2_w,self.f2_h))
        self.plot3.plotting_frame.grid(row=2, column=0, pady=10, padx=10, columnspan=3)
        
        self.act_pop = tk.StringVar(value='0')
        self.act_stims = tk.StringVar(value='all')
        self.act_trials = tk.StringVar(value='all')
        self.act_full = tk.StringVar(value='None')
        self.act_group = tk.IntVar(value=1)

        b = self.plot3.add_button('PLOT ACTIVITY', self.plot_act, "BW.TButton",)
        b.grid(row=0,column=0)

        l = self.plot3.add_label('pop', "BW.TLabel")
        l = l.grid(row=0,column=1,sticky='w')
        self.act_pop_m = self.plot3.add_menu(self.act_pop, [])
        self.act_pop_m.grid(row=0,column=2,sticky='w')

        l = self.plot3.add_label('stimuli', "BW.TLabel",)
        l.grid(row=0,column=3,sticky='w')
        self.act_stims_m = self.plot3.add_menu(self.act_stims, [])
        self.act_stims_m.grid(row=0,column=4,sticky='w')

        l = self.plot3.add_label('trials', "BW.TLabel",)
        l.grid(row=0,column=5,sticky='w')
        self.act_trials_m = self.plot3.add_menu(self.act_trials, [])
        self.act_trials_m.grid(row=0,column=6,sticky='w')

        l = self.plot3.add_label('single trials', "BW.TLabel",)
        l.grid(row=0,column=7,sticky='w')
        self.act_full_m = self.plot3.add_menu(self.act_full, ['None', 'None', 'norm', 'raw'])
        self.act_full_m.grid(row=0,column=8,sticky='w')

        l = self.plot3.add_checkbutton('group trials', self.act_group,)
        l.grid(row=0,column=9,sticky='w')

        self.plot3.update()

        # BIG PLOT 2#
        self.plot4 = PlotSection(master, figsize=(self.f3_w,self.f3_h))
        self.plot4.plotting_frame.grid(row=3, column=0, pady=1, padx=10, columnspan=3)

        self.plot4.update()

        ### 
        self.root.update()
        # self.root.bind("<Configure>", self.on_resize)

    def plot_pops(self):

        # plot clusters #
        embed_algo = self.pops_embed.get()
        pops_k = self.pops_k.get()
        clust_algo = self.pops_clust.get()
        fig = self.record.get_populations(algo=embed_algo,
                                        clusters=clust_algo,
                                        k=int(pops_k))[1]

        fig.set_size_inches(self.f1_w, self.f1_h)
        # resize the fonts
        for text in fig.findobj(match=plt.Text):
            text.set_fontsize(7)

        fig.tight_layout()
        self.plot1.update(fig)

        self.update_menu_args()

    def plot_fov(self):

        # plot fov #
        pops = self.fov_pops.get()
        bright = self.fov_bright_f.get()

        if pops == 'None':
            cells = None
        elif pops == 'all':
            cells  =self.record.responsive
        else:
            cells = self.record.populations[int(pops)]

        if bright != 'None': 
            bright = int(bright)
        else:
            bright = None
        
        img = plot_FOV(rec=self.record,
                       cells_ids=cells,
                       k = bright)
        
        fig, ax = plt.subplots(1,1, figsize=(self.f1_w, self.f1_h))
        ax.imshow(img, aspect='auto')
        ax.axis("off")
        fig.tight_layout()

        # rebuild the canvas
        self.plot2.update(fig)
        
    def plot_act(self):
        
        pop = int(self.act_pop.get())
        stimuli = self.act_stims.get()
        trials = self.act_trials.get()
        full = self.act_full.get()
        group = self.act_group.get()

        cells_pop = [self.record.cells[c] for c in self.record.populations[pop]]

        if stimuli == 'all':
            stimuli = self.record.sync.stims_names.copy()
        else: stimuli = [stimuli]
        if trials == 'all':
            trials = list(self.record.sync.trials_names.values())
        else: trials = [trials]

        # populate stim dict
        stim_dict = {s:[] for s in stimuli}
        for s in stimuli:
            for t in trials:
                if t in self.record.sync.stim_dict[s]:
                    stim_dict[s].append(t)

        if full == 'None': full = None

        # generate plot
        fig = plot_multipleStim(cells_pop, 
                                 stim_dict,
                                 plot_stim=True, 
                                 legend=False,
                                 ylabel='df/f',
                                 group_trials=group,
                                 full=full,
                                 dataBehav=self.record.dataBehav_analyzed)
        
        fig.set_size_inches(self.f2_w, self.f2_h)

        # resize the fonts
        for text in fig.findobj(match=plt.Text):
            text.set_fontsize(7)

        fig.tight_layout()
        self.plot3.update(fig)

        # generate heatmaps
        fig = plot_heatmaps(cells_pop, 
                            stim_dict,
                            normalize='zscore',
                            cbar=False)
        
        fig.set_size_inches(self.f3_w, self.f3_h)

        # resize the fonts
        for text in fig.findobj(match=plt.Text):
            text.set_fontsize(7)

        fig.tight_layout()
        self.plot4.update(fig)

    def update_menu_args(self):

        # update argumants from plots menus #
        def _update_menu_args(menu, var, args):

            menu.configure(state='normal')
            menu_ = menu['menu']
            menu_.delete(0, 'end')
            for arg_name in args:
                menu_.add_command(label=arg_name, command=tk._setit(var, arg_name))

        # update available populations #
        pops = np.arange(len(self.record.populations))
        # pops fov
        pops_fov = pops.copy().tolist()
        pops_fov.append('all')
        _update_menu_args(self.fov_pops_m, self.fov_pops, pops_fov)
        # pops act
        _update_menu_args(self.act_pop_m, self.act_pop, pops)

        # update stims
        stims = self.record.sync.stims_names.copy()
        stims.append('all')
        _update_menu_args(self.act_stims_m, self.act_stims, stims)
        # update trials
        trials = list(self.record.sync.trials_names.values())
        trials.append('all')
        _update_menu_args(self.act_trials_m, self.act_trials, trials)

    def run(self, input):
        
        nrecs = len(input['data_paths'])
        if len(input['behavior_paths']) != nrecs:
            input['behavior_paths'].extend(
            [None]*(nrecs-len(input)['behavior_paths']))
            
        loaders = []

        print('\n########################')
        print('##### LOADING DATA #####')
        print('########################\n')
        
        for datapath,syncpath,stimdictpath,behavpath in \
            zip(*input.values()):
            
            ld = loader.Dataloader()
            ld.load_sync(Path(syncpath), Path(stimdictpath), {0:'IPSI',1:'CONTRA',2:'BOTH'})
            ld.load_s2p_dir(str(Path(datapath)))
            if behavpath != None:
                ld.load_behav(Path(behavpath),Path(behavpath).stem)
            loaders.append(ld)

        if nrecs > 1:
            print('\n##################################')
            print('##### RUNNING BATCH ANALYSIS #####')
            print('##################################\n')
            record = batch.Batch(loaders)
            record.extract_all(keep_unresponsive=False)

        else:
            print('\n############################')
            print('##### RUNNING ANALYSIS #####')
            print('############################\n')  
            record = rec.Rec(loaders[0])
            record.extract_all(keep_unresponsive=False)

        self.record = record
        print('\n##################')
        print('##### DONE ! #####')
        print('##################\n')  

    # def on_resize(self,event):
        
    #     # determine the ratio of old width/height to new width/height
    #     wscale = float(event.width)/self.width
    #     hscale = float(event.height)/self.height
    #     if wscale<0.5 or hscale<0.5:
    #         print(event.width, self.width, event.height, self.height)
            
    #         # resize the figures 
    #         self.f1_w = 4*wscale
    #         self.f1_h = 3*hscale
    #         self.f2_w = 15*wscale
    #         self.f2_h = 5*hscale
    #         self.f3_w = 15*wscale
    #         self.f3_h = 4*hscale
    #         fsize = 7*hscale
            
    #         for p,s in zip([self.plot1,self.plot2,self.plot3,self.plot4],
    #                     [(self.f1_w,self.f1_h),(self.f1_w,self.f1_h)
    #                         ,(self.f2_w,self.f2_h),(self.f3_w,self.f3_h)]):
    #             fig = p.canvas.figure
    #             fig.set_size_inches(s[0],s[1])
    #             # resize the fonts
    #             for text in fig.findobj(match=plt.Text):
    #                 text.set_fontsize(fsize)

    #             fig.tight_layout()
    #             p.update(fig)

    #         self.root.update()



def main():
    root = tk.Tk()
    root.configure(background='black')
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()