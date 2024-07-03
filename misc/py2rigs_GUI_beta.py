import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from functools import partial
import sys
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pathlib import Path
import threading
import numpy as np
from Py2Rigs import rec, sync, loader, batch
from Py2Rigs.plot import *

INPUT_LIST = ['data_paths','sync_paths','stimdict_paths','behavior_paths']
USAGE =  "Welcome :)\n"\
         "In this GUI you can use Py2Rigs in single rec mode or in batch mode.\n"\
         "For adding a recording, browse and select:\n"\
         "- a valid suite2p output directory containing all the suite2p outputs in npy formats;\n"\
         "- a valid sync file containing the onsets and ofsetts frame of your events or stimuli (.mat or .npy);\n"\
         "- a valid event dictionary containing the sequence of your events or stimuli (.json);\n"\
         "- [OPTIONAL] a valid behavioral time siries you want to allign and plot with your data (.npy).\n"\
         "After adding the required paths, you can add the recording.\n"\
         "You can add how many recordings you want!\n"\
         "When you are redy, RUN!!\n"\

class StdoutRedirector:

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END) 
    
    def flush(self):
        pass

class InputFrame:

    INPUT_LIST = ['data_paths','sync_paths','stimdict_paths','behavior_paths']

    def __init__(self, master, run_command):

        ttk.Style().configure('TFrame', background='black')
        ttk.Style().configure("BW.TLabel", foreground="white", background="black", font=("Arial", 10))
        ttk.Style().configure("BW.TButton", foreground="black", borderwidth=1, background="red", font=("Arial", 10))
        ttk.Style().configure("S.TLabel", foreground="white", background="black", font=("Arial", 14))
        ttk.Style().configure("S.TButton", foreground="black", borderwidth=1, background="red", font=("Arial", 14))

        self.master = master
        self.run_command = run_command
        self.init_loc = '/'

        # input section
        self.input_paths = {ptype:tk.StringVar() for ptype in self.INPUT_LIST}
        self.input_lists = {ptype:[] for ptype in self.INPUT_LIST}
        self.nrecs = 0

        self.input_frame = ttk.Frame(self.master, style='TFrame')
        label = ttk.Label(self.input_frame, text="INPUTS", style='S.TLabel')
        label.grid(row=0, column=0, padx=0,sticky='nw')
        
        for i,(path_name, path_label) in enumerate(zip(self.INPUT_LIST,["* Suite2p output:", 
                                                                   "* Sync File (.mat):", 
                                                                   "* Event Dictionary (.json):", 
                                                                   "  Behavior (.npy):"])):
            if path_label=='* Suite2p output:':
                command = partial(self.browseDir,path_name)
            else:
                command = partial(self.browseFiles,path_name)

            label = ttk.Label(self.input_frame, text=path_label, style='BW.TLabel')
            label.grid(row=i+1, column=0, padx=0, sticky="nsew")
            entry = ttk.Button(self.input_frame, text='browse ...', command=command)
            entry.grid(row=i+1, column=1, pady=5, padx=5, sticky='nsew')
        
        b1 = ttk.Button(self.input_frame, text="Add Recording", style='BW.TButton', command=lambda path=path_name: self.add_paths())
        b1.grid(row=2, column=2, padx=3, pady=5, sticky='nsew')
        b2 = ttk.Button(self.input_frame, text="CLEAR", style='BW.TButton', command=self.clear)
        b2.grid(row=3, column=2, pady=5, padx=3, sticky='nsew')

        b3 = ttk.Button(self.input_frame, text="RUN", style='BW.TButton', command=self.run_callback)
        b3.grid(row=4, column=2, columnspan=1, padx=3, pady=5, sticky='nsew')
        
        # Standard Output Window
        self.output_frame = ttk.Frame(self.master, style='TFrame')
        self.output_frame.grid(row=0, column=1, columnspan=1, sticky='nsew')

        self.output_text = tk.Text(self.output_frame, wrap="word", bg='black', fg='white', bd=0, width=70, height=15)
        self.output_text.grid(row=0, column=0)
        
        # show usage message
        self.output_text.insert(tk.END, USAGE)
        self.output_text.see(tk.END)
        
        self.output_text.bind("<MouseWheel>", self.scroll_text)

        sys.stdout = StdoutRedirector(self.output_text)
        sys.stderr = StdoutRedirector(self.output_text)
    
    def browseFiles(self, path_name):
        filename = filedialog.askopenfilename(initialdir = self.init_loc,
                                                title = "Select a File")  
        path_var = self.input_paths[path_name]
        path_var.set(filename)

    def browseDir(self, path_name):
        dirname = filedialog.askdirectory(initialdir = self.init_loc,
                                           title = "Select a Suite2p output directory")
        path_var = self.input_paths[path_name]
        path_var.set(dirname)
        self.init_loc = dirname

    def clear(self):

        for ptype in self.input_paths.values(): ptype.set('')  
        self.input_lists = {ptype:[] for ptype in self.INPUT_LIST}
        self.output_text.delete('1.0', tk.END)
        if self.nrecs !=0: self.nrecs -= 1
    
    def add_paths(self):

        for i,path_name in enumerate(self.input_paths):

            path_var = self.input_paths[path_name]
            # path = TEST_PATHS[i]
            path = path_var.get()

            if path=='':
                if path_name != 'behavior_paths'and path_name != 'data_paths':
                    return f'\nERROR: Please provide all the required paths before adding!'
                    # self.clear()
                else:
                    path = None

            self.input_lists[path_name].append(path)
            path_var.set("")

        self.output_text.insert(tk.END, f"\nAdded REC_%d to batch list.\n"%self.nrecs)
        self.output_text.see(tk.END)
        self.nrecs += 1

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
        self.f3_w = 5
        self.f3_h = 9

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
        
        self.pops_embed = tk.StringVar(value='pca')
        self.pops_k = tk.StringVar()
        self.pops_clust = tk.StringVar(value='kmeans')

        b = self.plot1.add_button('GET POPULATIONS', self.plot_pops, "BW.TButton",)
        b.grid(row=0,column=0)

        l = self.plot1.add_label('embeding', "BW.TLabel")
        l = l.grid(row=0,column=1,sticky='w') 
        m = self.plot1.add_menu(self.pops_embed, ["pca", "umap", "tsne"])
        m.grid(row=0,column=2,sticky='w')

        l = self.plot1.add_label('n_clusers', "BW.TLabel",)
        l.grid(row=0,column=3,sticky='w')
        m = self.plot1.add_menu(self.pops_k, ['2', "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        m.grid(row=0,column=4,sticky='w')

        l = self.plot1.add_label('clustering', "BW.TLabel",)
        l.grid(row=0,column=5,sticky='w')
        m = self.plot1.add_menu(self.pops_clust, [ "kmeans", "gmm"])
        m.grid(row=0,column=6,sticky='w')

        self.plot1.update()

        ###

        # SECOND PLOT #
        self.plot2 = PlotSection(master, figsize=(self.f1_w,self.f1_h))
        self.plot2.plotting_frame.grid(row=0, column=2, rowspan=2, pady=10, padx=10, sticky='nsew')
        
        self.fov_pops = tk.StringVar(value='all')
        self.fov_recs = tk.StringVar(value='0')

        b = self.plot2.add_button('PLOT FOV', self.plot_fov, "BW.TButton",)
        b.grid(row=0,column=0)

        l = self.plot2.add_label('rec', "BW.TLabel")
        l = l.grid(row=0,column=1,sticky='w')
        self.fov_recs_m = self.plot2.add_menu(self.fov_recs, [])
        self.fov_recs_m.grid(row=0,column=2,sticky='w')

        l = self.plot2.add_label('populations', "BW.TLabel")
        l = l.grid(row=0,column=3,sticky='w')
        self.fov_pops_m = self.plot2.add_menu(self.fov_pops, [])
        self.fov_pops_m.grid(row=0,column=4,sticky='w')

        l = self.plot2.add_label('brighness', "BW.TLabel")
        l.grid(row=0,column=5,sticky='w')
        self.fov_bright_f = self.plot2.add_entry()
        self.fov_bright_f.insert(0, 'None')
        self.fov_bright_f.grid(row=0,column=6,sticky='w')

        self.plot2.update()

        ###

        # BIG PLOT 1#
        self.plot3 = PlotSection(master, figsize=(self.f2_w,self.f2_h))
        self.plot3.plotting_frame.grid(row=2, column=0, pady=10, padx=10, columnspan=3)
        
        self.act_pop = tk.StringVar(value='0')
        self.act_recs = tk.StringVar(value='all')
        self.act_stims = tk.StringVar(value='all')
        self.act_trials = tk.StringVar(value='all')
        self.act_full = tk.StringVar(value='None')
        self.act_group = tk.IntVar(value=1)

        self.cell_index = -1
        self.old_pop = -1

        b_up = self.plot3.add_button('<<', self.change_cell_nxt, "BW.TButton")
        b_up.grid(row=0, column=0,sticky='e')
        b = self.plot3.add_button('PLOT ACTIVITY', self.plot_act, "BW.TButton",)
        b.grid(row=0,column=1)
        b_down = self.plot3.add_button('>>', self.change_cell_prev, "BW.TButton")
        b_down.grid(row=0, column=2,sticky='w')

        l = self.plot3.add_label('pop', "BW.TLabel")
        l = l.grid(row=0,column=3,sticky='e')
        self.act_pop_m = self.plot3.add_menu(self.act_pop, [])
        self.act_pop_m.grid(row=0,column=4,sticky='w')

        l = self.plot3.add_label('rec', "BW.TLabel",)
        l.grid(row=0,column=5,sticky='e')
        self.act_recs_m = self.plot3.add_menu(self.act_recs, [])
        self.act_recs_m.grid(row=0,column=6,sticky='w')

        l = self.plot3.add_label('stimuli', "BW.TLabel",)
        l.grid(row=0,column=7,sticky='e')
        self.act_stims_m = self.plot3.add_menu(self.act_stims, [])

        self.act_stims_m.grid(row=0,column=8,sticky='w')

        l = self.plot3.add_label('trials', "BW.TLabel",)
        l.grid(row=0,column=9,sticky='e')
        self.act_trials_m = self.plot3.add_menu(self.act_trials, [])
        self.act_trials_m.grid(row=0,column=10,sticky='w')

        l = self.plot3.add_label('all trials', "BW.TLabel",)
        l.grid(row=0,column=11,sticky='e')
        self.act_full_m = self.plot3.add_menu(self.act_full, ['None', 'None', 'norm', 'raw'])
        self.act_full_m.grid(row=0,column=12,sticky='w')

        l = self.plot3.add_checkbutton('group trials', self.act_group,)
        l.grid(row=0,column=13,sticky='e')

        self.plot3.update()

        # BIG PLOT 2#
        self.plot4 = PlotSection(master, figsize=(self.f3_w,self.f3_h))
        self.plot4.plotting_frame.grid(row=0, column=3, pady=10, padx=5, rowspan=3, sticky='s')

        self.plot4.update()

        ### 
        self.root.update()


        # self.root.bind("<Configure>", self.on_resize)
        # # Timer to ensure the resize is finished
        # self.resize_timer = None

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

        if bright != 'None': 
            bright = float(bright)
        else:
            bright = None
        
        if self.nrecs > 1:
            rec_id = self.fov_recs.get()
            rec = self.record.recs[int(rec_id)]
        else:
            rec = self.record

        if pops == 'None':
            cells = None
        elif pops == 'all':
            cells = rec.responsive
        else:
            cells = rec.populations[int(pops)]

        img = plot_FOV(rec=rec,
                       cells_ids=cells,
                       k = bright)
        
        fig, ax = plt.subplots(1,1, figsize=(self.f1_w, self.f1_h))
        ax.imshow(img, aspect='auto')
        ax.axis("off")
        fig.tight_layout()

        # rebuild the canvas
        self.plot2.update(fig)
    
    def change_cell_nxt(self):

        pop = int(self.act_pop.get())
        rec_id = self.act_recs.get()
        if rec_id == 'all':
            rec = self.record
        else:
            rec = self.record.recs[int(rec_id)]
        
        cells_pop = [rec.cells[c] for c in rec.populations[pop]]

        if self.cell_index < len(cells_pop):
            self.cell_index += 1
            self.plot_act(avg=False)

    def change_cell_prev(self):

        self.cell_index -= 1

        if self.cell_index >= 0:
            self.plot_act(avg=False)

    def plot_act(self, avg=True):
        
        pop = int(self.act_pop.get())
        rec_id = self.act_recs.get()
        stimuli = self.act_stims.get()
        trial = self.act_trials.get()
        full = self.act_full.get()
        group = self.act_group.get()
            
        if rec_id == 'all':
            rec = self.record
        else:
            rec = self.record.recs[int(rec_id)]
        
        cells_pop = [rec.cells[c] for c in rec.populations[pop]]

        if stimuli == 'all':
            stimuli = self.record.sync.stims_names.copy()
        else: stimuli = [stimuli]
        if trial == 'all':
            trials = list(self.record.sync.trials_names.values())
        else: trials = [trial]

        # populate stim dict
        stim_dict = {s:[] for s in stimuli}
        for s in stimuli:
            for t in trials:
                if t in self.record.sync.stim_dict[s]:
                    stim_dict[s].append(t)
        
        if full == 'None': full = None

        # decide what to plot
        if avg:
            cells = cells_pop
            self.cell_index = -1
        else:
            cells = cells_pop[self.cell_index]

        # generate plot
        fig = plot_multipleStim(cells, 
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
        # check if need to update
        if pop != self.old_pop:
        
            fig = plot_heatmaps(cells_pop, 
                                stim_dict,
                                normalize='zscore',
                                vmin=-3,
                                vmax=3,
                                cbar=False)
            
            fig.set_size_inches(self.f3_w, self.f3_h)

            # resize the fonts
            for text in fig.findobj(match=plt.Text):
                text.set_fontsize(7)

            fig.tight_layout()
            self.plot4.update(fig)

            self.old_pop = pop

    def update_menu_args(self):

        # update argumants from plots menus #
        def _update_menu_args(menu, var, args):

            menu.configure(state='normal')
            menu_ = menu['menu']
            menu_.delete(0, 'end')
            for arg_name in args:
                menu_.add_command(label=arg_name, command=tk._setit(var, arg_name))

        if self.nrecs > 1:
            # update recordings
            recs_id = list(self.record.recs.keys())
            _update_menu_args(self.fov_recs_m, self.fov_recs, recs_id)
            recs_id.append('all')
            _update_menu_args(self.act_recs_m, self.act_recs, recs_id)
            self.act_recs.set('all')

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
        
        self.nrecs = len(input['data_paths'])

        if len(input['behavior_paths']) != self.nrecs:
            input['behavior_paths'].extend(
            [None]*(self.nrecs-len(input)['behavior_paths']))
            
        loaders = []

        print('\n########################')
        print('##### LOADING DATA #####')
        print('########################\n')
        
        for datapath,syncpath,stimdictpath,behavpath in \
            zip(*input.values()):
            
            ld = loader.Dataloader()
            ld.load_sync(Path(syncpath), Path(stimdictpath))
            ld.load_s2p_dir(str(Path(datapath)))
            if behavpath != None:
                ld.load_behav(Path(behavpath),Path(behavpath).stem)
            loaders.append(ld)

        if self.nrecs > 1:
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

        # plot!
        self.plot_pops()
        self.plot_fov()
        self.plot_act()

    def on_resize(self, event):

        # Cancel any previously scheduled resize events
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        
        # Schedule the final resize event
        self.resize_timer = self.root.after(400, self.final_resize, event)
    
    def final_resize(self, event):
        # Get the new size
        width = event.width
        height = event.height
        print(width, height)
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width)/self.width*2
        hscale = float(event.height)/self.height*2
        # if wscale<0.5 or hscale<0.5:
            
        # resize the figures 
        self.f1_w = 4*wscale
        self.f1_h = 3*hscale
        self.f2_w = 15*wscale
        self.f2_h = 5*hscale
        self.f3_w = 15*wscale
        self.f3_h = 4*hscale
        fsize = 7*hscale
        
        for p,s in zip([self.plot1,self.plot2,self.plot3,self.plot4],
                    [(self.f1_w,self.f1_h),(self.f1_w,self.f1_h)
                        ,(self.f2_w,self.f2_h),(self.f3_w,self.f3_h)]):
                        
            fig = p.canvas.figure
            fig.set_size_inches(s[0],s[1])
            print(s[0],s[1])
            # resize the fonts
            for text in fig.findobj(match=plt.Text):
                text.set_fontsize(fsize)

            fig.tight_layout()
            p.update(fig)
            # p.canvas.draw()

        # self.root.update()

        

        
        
        


def main():
    import warnings

    warnings.simplefilter('ignore')
    root = tk.Tk()
    root.title('Py2Rigs')
    root.configure(background='black')
    root.resizable(True, True)
    gui = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()