import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib import cm,patches
from scipy.signal import convolve2d
import warnings

# from .core import *
from .sync import Sync
from .cell import Cell
from .plot_stims import *
from .utils import z_norm, lin_norm, check_len_consistency

warnings.simplefilter('always')

# TRIALS_C = {
#    0:  (0.08,0.6,0.3), #"#169b4e",
#    1:  (0.1,0.27,0.63), #"#1b46a2",
#    2:  (0.69,0.19,0.15), #"#b03028",
#    3:  (0.69,0.6,0.9), #"#b194e4",
#   }  # [IPSI, CONTRA, BOTH, ...]


cmap = cm.get_cmap('gist_rainbow')
rng = np.random.default_rng(seed=5)
index = rng.integers(0,cmap.N,100)
C_seq = [cmap(int(i)) for i in np.linspace(0,cmap.N,50)]
C_rand = [cmap(int(i)) for i in index]


TRIALS_C = [

    (0.576, 0.439, 0.859, 1),     # Medium Purple
    (1, 0.549, 0, 1),              # Dark Orange
    (0, 0.502, 0.502, 1),          # Teal
    (1, 0.078, 0.576, 1),          # Deep Pink
    (0, 1, 0.271, 1),              # Neon Green
    (1, 0, 0.502, 1),              # Raspberry
    (0.4, 0.804, 0.667, 1),        # Medium Aquamarine
    (1, 0.647, 0, 1),              # Orange
    (0, 0.502, 1, 1),              # Royal Blue
    (1, 0.412, 0.706, 1),          # Hot Pink
    (0.251, 0.878, 0.816, 1),      # Turquoise
    (1, 0.843, 0, 1),              # Gold
    (0.729, 0.333, 0.827, 1),      # Medium Orchid
    (0, 1, 1, 1),                  # Cyan
    (1, 0.271, 0, 1),              # Red-Orange
    (0.627, 0.125, 0.941, 1),      # Purple
    (0, 1, 0.498, 1),              # Spring Green
    (0.855, 0.667, 0.125, 1),      # Goldenrod
    (0.678, 0.847, 0.902, 1),      # Light Blue
    (1, 0.388, 0.278, 1),          # Tomato
    (0.294, 0, 0.51, 1),           # Indigo
    (0.678, 1, 0.184, 1),          # Green Yellow
    (1, 0.078, 0.576, 1),          # Deep Pink
    (0, 0.98, 0.604, 1),           # Medium Spring Green
    (1, 0.412, 0.706, 1),          # Hot Pink
    (1, 0.843, 0, 1),              # Gold
    (0, 0.98, 0.604, 1),           # Medium Spring Green
    (0.275, 0.51, 0.706, 1),       # Steel Blue
    (0.941, 0.902, 0.549, 1),      # Khaki
    (1, 0.412, 0.706, 1),          # Hot Pink
    (0.125, 0.698, 0.667, 1),      # Light Sea Green
    (1, 0.843, 0, 1),              # Gold
    (0.941, 0.502, 0.502, 1),      # Light Coral
    (0, 0.502, 0.502, 1),          # Teal
    (0.678, 1, 0.184, 1),          # Green Yellow
    (1, 0, 1, 1),                  # Magenta
    (0, 1, 1, 1),                  # Cyan
    (1, 0.271, 0, 1),              # Red-Orange
    (0.502, 0.502, 0, 1),          # Olive
    (0.294, 0, 0.51, 1),           # Indigo
    (0.729, 0.333, 0.827, 1),      # Medium Orchid
    (0.678, 1, 0.184, 1),          # Green Yellow
    (1, 0, 0.502, 1),              # Raspberry
    (1, 0.647, 0, 1),              # Orange
    (0.855, 0.667, 0.125, 1),      # Goldenrod
    (0, 0.98, 0.604, 1),           # Medium Spring Green
    (1, 0.271, 0, 1),              # Red-Orange
    (0.275, 0.51, 0.706, 1),       # Steel Blue
    (0.941, 0.902, 0.549, 1)       # Khaki
            ]
# POPS_C = list(colors.TABLEAU_COLORS.keys())
POPS_C  = [c for c in sns.color_palette('pastel')]

MARKERS = [
            "o",
            "v",
            "s",
            "P",
            "d",
            "1",
            "*"
        ]


### DRAWING FUNCTIONS ###

def draw_singleStim(
    ax,
    cells,
    stim,
    trials,
    rtype="norm",
    ylabel="",
    stim_window = True,
    color=None,
    legend = False,
    func_ = None):

    """
    Plot average response calculated across all the specified cells .

    - cells: 
        cells to plot. if a list of C2p object is passed, plot the mean. 
    - sync: Sync
        Sync object associated to the cells
    - stim:
        stimulation condition for which to plot the average
    - trials:
        trials to plot, can be str or list of str. If None, use all the possible trials 
    -rtype: str
        can be "norm", "raw"
    - func: dict
        function that will be applied to each signal. first argoument of the function is 
        is assumed to be the signal. Should be in the form:
        (func,**kwards) where kwards should not contain first argoument (signal)
    """

    if isinstance(trials, str):

        trials = [trials]

    if isinstance(cells, list):

        resp_dict = {}

        for trial in trials:


            resp_dict |= {trial:{}}

            cells_avgs = []

            for cell in cells:

                r = cell.analyzed_trials[stim][trial]["%s_avg"%rtype]

                cells_avgs.append(r)

            # check for lenght consistency
            cells_avgs = check_len_consistency(cells_avgs)

            # apply the function
            if func_ != None:

                cells_avgs = func_[0](cells_avgs,**func_[1])

            resp_dict[trial] |= {"mean":np.mean(cells_avgs,axis=0), 
                                "sem":np.std(cells_avgs,axis=0)/np.sqrt(len(cells_avgs))}
    else:

        cell = cells

    # invert trials dict
    inverted_trials_dict = {v:k for k,v in cell.sync.trials_names.items()}

    ymax = 0
    ymin = 0

    # draw the traces
    title = stim+" -"
    for trial in trials:
        
        if isinstance(cells, list):

            r = resp_dict[trial]["mean"]
            error = resp_dict[trial]["sem"]

        else: 

            if cells != None:
            
                r = cell.analyzed_trials[stim][trial]["%s_avg"%rtype]
                error = cell.analyzed_trials[stim][trial]["%s_std"%rtype]

            if func_ != None:

                r = func_[0](r,**func_[1])
                error = func_[0](error,**func_[1])

        if (r + error).max() > ymax:

            ymax = (r + error).max()

        if (r - error).min() < ymin:

            ymin = (r - error).min() 

        sf = cell.rec.sf
        
        x = np.arange(len(r)) / sf

        on_frame = (
            cell.analyzed_trials[stim][trial]["window"][0] / sf
        )
        off_frame = (
            cell.analyzed_trials[stim][trial]["window"][1] / sf
        )

        if stim_window:

            ax.axvspan(on_frame, off_frame, color="k", alpha=0.2)
            # only draw stim window for first trial
            stim_window = False

            ax.tick_params(labelsize=35)
            ax.set_xticks(
                [on_frame, off_frame]
            )

            ax.set_xticklabels(
                [0, round(off_frame-on_frame,1)]
            )

        if color == None:

            c = TRIALS_C[inverted_trials_dict[trial]]

        else: c=color

        ax.plot(x, r, c=c, linewidth=1.4, alpha=0.6, label=trial)

        if isinstance(error,np.ndarray):

            ax.fill_between(
                x, r + error, r - error, color=np.array(c)*0.5, alpha=0.3
            )

        title += " %s"%trial

        if ylabel==None:
        
            if rtype == "norm":

                ax.set_ylabel("\u0394F/F",fontsize=18)
            
            if rtype == "spks":
                
                ax.set_ylabel("OASIS spikes",fontsize=18)

            if rtype == "zspks":
                
                ax.set_ylabel("OASIS spikes (z-score)",fontsize=18)

        ax.set_ylabel(ylabel,fontsize=35)

        ax.set_xlabel("time [s]",fontsize=35)

        if legend:

            ax.legend(fontsize="15")

        # ax.set_title(title,fontsize=18)

    ax.spines[['right','top','left','bottom']].set_visible(False)
    # ax.grid('dashed',linewidth = 1.5, alpha = 0.25)

    return ax, ymax, ymin

def draw_singleStim_dataBehav(
    ax,
    dataBehav,
    stim,
    trials,
    trials_names_dict=None,
    sf=15.5,
    rtype="norm",
    ylabel=None,
    stim_window = True,
    color=(0,0,0),
    legend = False,
    func_ = None):
    
    """
    Plot average response calculated across all the specified cells .

    - dataBehav: dict
        dataBehav dict from a rec obj. 
    - sync: Sync
        Sync object associated to the cells
    - stim:
        stimulation condition for which to plot the average
    - trials:
        trials to plot, can be str or list of str. If None, use all the possible trials ,
    trials_names_dict: dict
        in the form of {trial_id : trial_name}, if specified, color code the trials
    -rtype: str
        can be "norm", "raw"
    - func: dict
        function that will be applied to each signal. first argoument of the function is 
        is assumed to be the signal. Should be in the form:
        (func,**kwards) where kwards should not contain first argoument (signal)
    """
        
    if isinstance(trials, str):

        trials = [trials]

    ymax = 0
    ymin = 0

    name = list(dataBehav.keys())[0]
    dataBehav = dataBehav[name]

    # draw the traces
    title = stim+" -"
    for i,trial in enumerate(trials):

        r = dataBehav[stim][trial]["%s_avg"%rtype]
        error = dataBehav[stim][trial]["%s_err"%rtype]

        if func_ != None:

            r = func_[0](r,**func_[1])
            error = func_[0](error,**func_[1])

        if (r + error).max() > ymax:

            ymax = (r + error).max()

        if (r - error).min() < ymin:

            ymin = (r - error).min() 
        
        x = np.arange(len(r)) / sf

        on_frame = (
            dataBehav[stim][trial]["window"][0] / sf
        )
        off_frame = (
            dataBehav[stim][trial]["window"][1] / sf
        )

        if stim_window:

            ax.axvspan(on_frame, off_frame, color="k", alpha=0.2)
            # only draw stim window for first trial
            stim_window = False

            ax.tick_params(labelsize=35)
            ax.set_xticks(
                [0, on_frame, on_frame+1, off_frame, round(len(r) / sf)]
            )

            ax.set_xticklabels(
                [round(-on_frame,1), 0, 1, round(off_frame-on_frame,1), round(len(r) / sf,1)]
            )

        if trials_names_dict != None:
            color = TRIALS_C[trials_names_dict[trial]]

        ax.plot(x, r, c=color, linewidth=2, alpha=0.6, label=trial)

        if isinstance(error,np.ndarray):

            ax.fill_between(
                x, r + error, r - error, color=np.array(color)*0.8, alpha=0.3
            )

        title += " %s"%trial

        if ylabel==None:
        
            if rtype == "norm":

                ylabel = "%s [norm]"%name

            elif rtype == "raw":

                ylabel = "%s"%name

        ax.set_ylabel(ylabel,fontsize=35)

        ax.set_xlabel("time [s]",fontsize=35)

        if legend:

            ax.legend()

        # ax.set_title(title,fontsize=18)
    # ax.set_ylim(-3,1)

    ax.spines[['right','top','left','bottom']].set_visible(False)
    # ax.grid('dashed',linewidth = 1.5, alpha = 0.25)

    return ax, ymax, ymin
        
def draw_full(
    ax,
    cells,
    sync: Sync,
    rtype="norm",
    func_=None,
    stim=None,
    ylabel=None,
    x_scale='sec'):

    """
    Plot full length traces for all cells. If stim is specified, plot full trace
    for that stim. If cells is a list of C2p object, plot the average.

    - rtype: (str)
        type of response, can be either "raw" or "norm" 

    """

    # invert trials dict
    inverted_trials_dict = {v:k for k,v in sync.trials_names.items()}

    if isinstance(cells, list):

        avg = []

        for cell in cells:
            avg.append(eval("cell."+rtype))
        
        avg = check_len_consistency(avg)
        r = np.mean(avg, axis=0)

    else:

        r = eval("cells."+rtype)
        cell = cells


    if stim != None:

        r = r[sync.sync_ds[stim]["stim_window"][0]-
              int(cell.params["pre_trial"]*cell.rec.sf):       ###
              (sync.sync_ds[stim]["stim_window"][1]+
              int(cell.params["baseline_length"]*cell.rec.sf))]
        
        if func_ != None:
            r = func_[0](r,**func_[1])

        stims = [stim]
        offset = sync.sync_ds[stim]["stim_window"][0]-(cell.params["pre_trial"]*cell.rec.sf)
        
    else:

        stims = sync.sync_ds
        offset = 0

    if x_scale=='sec':

        ax.set_xlabel("time [s]", fontsize=18)

        if rtype == 'norm' or  rtype == 'raw':  
            sf = cell.params["sf"]
            # linewidth=0.1

    elif x_scale=='samples':

        ax.set_xlabel("samples", fontsize=18)
        sf = 1

    x = np.arange(len(r))/sf

    ax.set_title(rtype)

    ax.plot(x, r, c='k',linewidth=1,alpha=0.8)

    ax.set_xlabel("time [s]")

    if ylabel != None:

        ax.set_ylabel(ylabel, fontsize=25)

    for stim in stims:

        for trial_type in list(sync.sync_ds[stim].keys())[:-1]:

            c = TRIALS_C[inverted_trials_dict[trial_type]]

            for i,trial in enumerate(sync.sync_ds[stim][trial_type]["trials"]):

                ax.axvspan(int((trial[0]-offset)/sf), int((trial[1]-offset)/sf),
                                color=c, alpha=0.2, label="_"*i+"%s -%s"%(stim,trial_type))

    ax.tick_params(labelsize=20)            
    ax.spines[['top', 'right']].set_visible(False)         
    ymax = r.max()
    ymin = r.min()

    return ax, ymax, ymin

def draw_events_full(
    ax,
    cell,
    sync,
    stim,
    rtype='raw',
    thresh=0,
    ylabel=None,):

    """
    Plot full length rester plot for all cells. If stim is specified, plot full trace
    for that stim. If cells is a list of C2p object, plot the average.

    - rtype: (str)
        type of response, can be either "raw" or "norm" 

    """

    # invert trials dict
    inverted_trials_dict = {v:k for k,v in sync.trials_names.items()}

    ax.set_xlabel("time [s]", fontsize=18)

    if (rtype == 'norm' or  rtype == 'raw'):  
        sf = cell.rec.sf

    all_trials = cell.analyzed_trials[stim]
    
    events = []
    offsets = []

    for i,trial in enumerate(all_trials):

        n_trials = len(all_trials[trial]['trials_%s'%rtype])
        events.extend([(np.where(r>thresh)[0]/sf).tolist() for r in all_trials[trial]['trials_%s'%rtype]])
        off = [5*i+j*4/n_trials for j in range(n_trials)]
        offsets.extend(off)
        ax.axhspan(off[0]-0.5, off[-1]+0.5, color=TRIALS_C[inverted_trials_dict[trial]], alpha=0.3)

    ax.eventplot(events, lineoffsets=offsets, linelengths=[4/n_trials*0.7]*len(offsets), color='k', linewidth=0.4)
    
    # draw stim window
    on = cell.params['pre_trial']
    off = (on+sync.sync_ds[stim][trial]['trial_len']/sf)
    ax.axvspan(on, off, color='k', alpha=0.17)

    ax.set_title(rtype)
    ax.set_xlabel("time [s]")
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_yticks([])

    if ylabel != None:

        ax.set_ylabel(ylabel, fontsize=25)

    return ax

def draw_heatmap(
        matrix, 
        vmin=None, 
        vmax=None,
        cbar=False,
        cb_label="", 
        ax=None):

    """
    Base function for drawing an hetmap from input matrix.
    """

    m = np.array(matrix)

    map = sns.diverging_palette(300, 120, s=80, l=80, n=200, as_cmap=True,)
    #sns.diverging_palette(120, 230, s=60, l=60, n=200, as_cmap=True,)
    #sns.diverging_palette(145, 300, s=60, as_cmap=True)


    ax = sns.heatmap(
        m,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        ax=ax,
        cbar=cbar,
        cmap=map,
        cbar_kws=dict(
        use_gridspec=False, location="bottom", pad=0.05, label=cb_label, aspect=60
        ),
    )

    ax.set_yticks(list(range(100,matrix.shape[0],100)),list(range(100,matrix.shape[0],100)),fontsize=35)
    ax.figure.axes[-1].xaxis.label.set_size(25)

    return ax

### PLOTTING FUNCTIONS ###


def plot_multipleStim(
    cells: list,
    stim_dict,
    average=True,
    rtype="norm",
    func_=None,
    full="norm",
    dataBehav=None,
    stim_window=True,
    ylabel='',
    ylim=None,
    qi_threshold=0,
    nmax=10,
    plot_stim =True,
    order_stims=False,
    group_trials=True,
    share_x=False,
    share_y=False,
    legend=False,
    save=False,
    save_path="",
    save_suffix="",):

    """
    Plot the responses for thwe specified cells. If average is True, plot the population average.
    If not, plot each cell independently. You can also specify the stimuli and the trialrtypes you
    want to plot.

    - cells: list
        list of C2p objects
    - stim_dict: dict
        dict containing stim:[trials] items specyfying what to plot
    - average: bool
        wether to compute the population average or not
    - stims_names:
        stimulation conditions for which to plot the average
    - trials_names:
        trials to plot
    -rtype: str
        can be "raw", "norm", "spks"  or "zspks"
    - full : str
        can be "norm" or "zspks", Fraw or Fneu. If None, full trace will not be plotted
        WARNING: 
        Don't plot averaged full traces of populations of cells from different recordings,
        as the stimulation pattern is likely to be different for each recording!
    - dataBehav: dict
        pair of name:data of alligned behavioral data to plot
    - order_stims: bool
        wether to order the stimuli subplots or not. the ordering is based on the name.
    - func_: tuple
        function that will be applied to each signal. first argoument of the function is 
        is assumed to be the signal. Should be in the form:
        (func,**kwards) where kwards should not contain first argoument (signal)
    """

    # fix the arguments

    if not isinstance(save_path, list):

        save_path = [save_path]

    stims = list(stim_dict.keys())
    stims.sort()
    trials = list(set([t for stim in stim_dict for t in stim_dict[stim]]))

    if order_stims:

        stims = sorted(stims)

    ## convert to list if single elements
    elif not isinstance(stims, list): 
        
        stims = [stims]

    n_stims = len(stims)
    n_trials = len(trials)

    ## decide wheter to plot all the cells or the average
    if not isinstance(cells, list):

        average = False
        cells = [cells]

    elif average:

        cells = [cells]

    ## if full is specidied, and average==False, add a row to the figure
    add_full = 0
    add_stim = 0
    add_behav = 0

    if full != None:

        add_full = 1

    if dataBehav != None:
        
        add_behav = len(dataBehav)

    for stim in stims:

        stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
        if plot_stim and stim_func in globals():

            add_stim = 1

    if add_stim == 0:
        plot_stim = False

    add_row = add_full+add_stim+add_behav

    # main loop
    for i,c in enumerate(cells):

        if group_trials:
            figsize = (15*n_stims, 4*(2+add_row))
            nrows = (1+add_row)
        else:
            figsize = (9*n_stims, 8*n_trials+add_row)
            nrows=(n_trials+add_row)

        height_ratios = [1]*nrows

        if plot_stim:
            height_ratios[0] /= 3
                       
        fig, axs = plt.subplots(nrows=nrows, ncols=n_stims, figsize=figsize,
                                gridspec_kw={'height_ratios': height_ratios})

        if average:

            fig.suptitle("Population Average - %d ROIs"%len(c),fontsize=18)

            # retrive syncs
            sync = c[0].sync
            # spikes?
            spks = (c[0].rec.dtype=='spikes')

        else:

            if i >= nmax: 
                plt.close(fig)
                break

            if c.qi != None:
                qi = str(round(c.qi,2))
            else:
                qi = "nd"
            
            fig.suptitle("ROI #%s   QI:%s"%(c.id,qi),fontsize=18)

            # retrive syncs
            sync = c.sync
            # spikes?
            spks = (c.rec.dtype=='spikes')

            # plot only the cells with qi above qi_threshold
            if c.qi != None and c.qi < qi_threshold:
                plt.close(fig)
                continue

        if not isinstance(axs, np.ndarray):

            axs = np.array([axs])

        y_max = 0
        y_min = 0
        y_max_behav = 0
        y_min_behav = 0

        for j,stim in enumerate(stims):

            # y_max = 0
            # y_min = 0

            trials_ = sorted(stim_dict[stim])
            
            # PLOT GRUPPED TRIALS
            if group_trials:

                ## make sure to use only trials which exist for a specific stim
                # trials_ = set(trials).intersection(set(sync.sync_ds[stim]))
                # trials_ = sorted(stim_dict[stim])

                axs_ = axs

                if full != None:

                    axs_ = axs[0]

                if plot_stim:

                    axs_ = axs[1]

                else:

                    axs_ = axs_

                if len(stims)>1:

                    axs_ = axs_[j]

                elif not plot_stim:

                    if isinstance(axs_,np.ndarray): ##

                        axs_ = axs_[0]

                _, ymax, ymin = draw_singleStim(axs_, c, stim, trials_, rtype, ylabel=ylabel, 
                                                stim_window=stim_window,func_=func_, legend=legend)

                if ymax > y_max: y_max = ymax

                if ymin < y_min: y_min = ymin

                # if j>0: axs_.set_ylabel("")

                if full or dataBehav:

                    axs_.set_xlabel("")

            else:

                # PLOT UNGRUPPED TRIALS
                axs_T = axs.T

                if len(stims)>1:

                    axs_T = axs_T[j]

                trials_ = sorted(stim_dict[stim])

                for i,trial in enumerate(trials_, add_stim):

                    ## make sure to use only trials which exist for a specific stim
                    if trial in sync.sync_ds[stim]:
                        
                        _, ymax, ymin = draw_singleStim(axs_T[i], c, stim, trial, rtype, ylabel=ylabel,
                                                        stim_window=stim_window, func_=func_, legend=legend)

                        if ymax > y_max: y_max = ymax

                        if ymin < y_min: y_min = ymin

                        if i>0: axs_T[i].set_title("")

                    else:
                        
                        axs_T[i].axis("off")
                        axs_T[i].set_title("")

                        if i>0: axs_T[i].set_title("")

                    if i < len(trials_) or add_full or dataBehav:
                        axs_T[i].set_xlabel("")
                
            # plot dataBehav if desired
            if dataBehav != None:

                ndbehav = len(dataBehav)

                for ix,(name,dbehav) in enumerate(dataBehav.items(),1):
                
                    if axs.ndim==1:
                        
                        ax_behav = axs[(-ix-add_full)]

                    else:

                        ax_behav = axs[(-ix-add_full),j]

                    if not isinstance(c,list):

                        trials_names_dict = {v:k for k,v in c.sync.trials_names.items()}
                    else: 
                        trials_names_dict = {v:k for k,v in c[0].sync.trials_names.items()}

                    _, ymax_behav, ymin_behav = draw_singleStim_dataBehav(ax_behav, {name:dbehav}, stim, stim_dict[stim], 
                                                                            rtype='norm', trials_names_dict=trials_names_dict,
                                                                            ylabel='% change')

                    if ymax_behav > y_max_behav: y_max_behav = ymax_behav

                    if ymin_behav < y_min_behav: y_min_behav = ymin_behav

                    if full:

                        ax_behav.set_xlabel("")
            
                    # ax_behav.set_ylim(0.2,0.8)

            if full != None:
                
                if axs.ndim==1:

                    ax_full = axs[-1]

                else:

                    ax_full = axs[-1,j]

                if full == 'norm' or full == 'raw': 
                    ylabel_full = ylabel
                else:
                    ylabel_full = ylabel

                if spks and isinstance(c, Cell):

                    _ = draw_events_full(ax_full, c, sync, rtype=full, stim=stim)
                else:
                    _, ymax, ymin = draw_full(ax_full, c, sync, rtype=full, func_=None, stim=stim, ylabel=ylabel_full)

                # if ymax > y_max: y_max = ymax

                # if ymin < y_min: y_min = ymin

                ax_full.set_title("")

            if plot_stim:

                stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
                axs_T = axs.T

                if len(stims)>1:

                    axs_T = axs_T[j]
                
                ## retrive parameters 
                cell = c
                if isinstance(c, list):
                    cell = c[0]

                sf = cell.rec.sf
                pad_l = round(cell.params["pre_trial"]*cell.rec.sf)

                frames_tot = 0
                for ax in axs_T[1:]:
                    if ax.lines:
                        frames_tot = ax.lines[0].get_xdata().size
                        break       

                stim_l = cell.sync.sync_ds[stim][trials_[0]]['trial_len']

                if stim_func in globals() and False:

                    func = globals()[stim_func]
                    func(frames_tot=int(frames_tot), stim_l=stim_l, ax=axs_T[0], pad_l=pad_l, fr=sf)

                else:
                    general_stim(frames_tot=frames_tot, stim_l=stim_l, ax=axs_T[0], pad_l=pad_l)

                axs_T[0].axis('off')
                axs_T[0].set_title(stim, fontsize=22)  

        # set axis lim
        axs_neu = axs[add_stim:-(add_full+add_behav)]
    
        for ax in axs_neu.flatten():
            
            if ylim != None:
                ax.set_ylim(ylim[0],ylim[1])
            
            else:
                ax.set_ylim(y_min-(abs(y_min/10)), y_max+(abs(y_max/10)))
        
        if add_behav:
            axs_behav = axs[-(add_full+add_behav)]

            for ax in axs_behav.flatten():
                ax.set_ylim(y_min_behav-(abs(y_min_behav/10))-1, y_max_behav+(abs(y_max_behav/10))+1)


        if share_y:
            plt.subplots_adjust(wspace=0.01)

        if share_x:
            plt.subplots_adjust( hspace=0.01)
            
        if save:

            if average:

                for path in save_path:

                    plt.savefig(r"%s/pop_avg_%s%s.png" %(path,rtype,save_suffix),
                                bbox_inches="tight")

            else:

                for path in save_path:
                
                    plt.savefig(r"%s/ROI_#%s_%s%s.png" %(path,c.id,rtype,save_suffix),
                                bbox_inches="tight")
        # plt.close(fig)
        return fig
    
def plot_FOV(
        rec,
        cells_ids,
        save_path=None, 
        k=None, 
        img="meanImg"):

    '''
    Plot mean image of the FOV with masks of the passed cells.
    If cells_ids is list of lists, each sublist will be considered as a population.

    - rec: R2p
        R2p object from which the cells have been extracted.
    - cells: list 
        list of valid cells ids
    - k: int
        value to scale the image luminance. 
        If None, the brightness is automatically adjusted
    - img: str
        a valid name of an image stored in stat.npy

    '''

    mean_img = rec.ops[img]

    img_rgb = ((np.stack([mean_img,mean_img,mean_img],axis=2)-np.min(mean_img))/np.max((mean_img-np.min(mean_img))))
    if  k == None : k = 0.22/np.mean(img_rgb)
    img_rgb = img_rgb*k

    fig,ax = plt.subplots(1,1,figsize=(10,7))
    # ax.imshow(img_rgb, aspect='auto')

    all_labels = []

    if cells_ids != None:
        # for i,pop in enumerate(cells_ids):
        for idx in cells_ids:

            # c = POPS_C[i]
            pop = rec.cells[idx].label
            all_labels.append(pop)
            c = POPS_C[pop]

            # extract and color ROIs pixels

            ypix = rec.stat[idx]['ypix']
            xpix = rec.stat[idx]['xpix']

            # make contour
            mask = np.zeros_like(mean_img)
            mask[ypix,xpix] = 1
            mask_diff = np.abs(np.diff(mask,append=mask[-1,-1],axis=0))+np.abs(
                                np.diff(mask,append=mask.T[-1,-1],axis=1))
            # increase border width
            mask_diff = convolve2d(mask_diff,[[1,1],[1,1]])
            contour_pix = np.where(mask_diff!=0)

            img_rgb[contour_pix] = colors.to_rgb(c)

        for pop in set(all_labels):

            ax.plot(0,0,c=POPS_C[pop],label='POP_#%d'%pop)
            leg = plt.legend(loc="upper right", bbox_to_anchor=(1.16, 1.0),facecolor='white')
            leg.get_frame().set_linewidth(0.0)

    ax.imshow(img_rgb, aspect='auto')
    plt.axis('off')

    if save_path != None:
        plt.savefig(save_path, bbox_inches="tight")

    return img_rgb
  
def plot_heatmaps(
    cells,
    stim_dict,
    rtype="norm",
    full=None,
    vmin=None,
    vmax=None,
    normalize=None,
    plot_stim=False,
    save=False,
    save_path="",
    name="",
    cbar=False,
    cb_label="",):

    """
    Plot heatmap for all the cells.

    - cells: list
        list of C2p objects
    - stim_dict: dict
        dict containing stim:[trials] items specyfying what to plot
    -rtype: str
        can be "norm" or "zspks"
    - full: str
        can be Fraw,norm,spks or zspks. if specified, stims,trials andrtype
        argoument will be ignored
    - normalize: str
        can be 'zscore' or 'linear' if specified
    """

    
    # plot averages
    if full == None:

        stims = list(stim_dict.keys())
        stims.sort()

        all_trials = []

        for stim in stim_dict:
            all_trials.extend(stim_dict[stim])

        all_trials = np.unique(all_trials)
        

        add_row = 0
        for stim in stims:

            stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
            if plot_stim and (stim_func in globals()):
                add_row=1

        nrows = len(all_trials)+add_row
        height_ratios = [1]*nrows

        if add_row:
            height_ratios[0] /= 2
                       
        fig, axs = plt.subplots(nrows=nrows, ncols=len(stims),figsize=(15*len(stims),6*nrows), 
                                gridspec_kw={'height_ratios': height_ratios})

        if cbar:
            cbar_ax = fig.add_axes([.3, .05, .4, .015])

        for i,trial in enumerate(all_trials):

            i = i+add_row
            
            for j,stim in enumerate(stims):

                if not isinstance(axs, np.ndarray):
                    ax = axs

                elif len(stims)==1:
                    ax=axs[i]

                elif len(all_trials)==1:
                    ax=axs[j]

                else:
                    ax=axs[i,j]

                if trial in stim_dict[stim]:
                
                    # extract data from each cell
                    resp_all = []
                    for cell in cells:

                        r = cell.analyzed_trials[stim][trial]["%s_avg"%rtype]
                        resp_all.append(r)

                    resp_all = check_len_consistency(resp_all)
                    # convert to array
                    resp_all = np.array(resp_all)

                    # normalize
                    if normalize == "linear":

                        resp_all = lin_norm(resp_all)

                    elif normalize == "zscore":

                        resp_all = z_norm(resp_all)

                    # sort matrix according to quality index
                    qis = [cell.qi for cell in cells]
                    qis_sort = np.argsort(qis)
                    resp_all = resp_all[np.flip(qis_sort)]

                    # plot
                    # if i==len(all_trials):
                    #     cbar=True

                    draw_heatmap(resp_all,vmin=vmin,vmax=vmax,cb_label=cb_label,cbar=False,ax=ax)   
                    ax.axvline(cell.analyzed_trials[stim][trial]['window'][0],0,color='r',linestyle='--',alpha=0.8) 
                    ax.axvline(cell.analyzed_trials[stim][trial]['window'][1],0,color='r',linestyle='--',alpha=0.8) 

                    if isinstance(axs,np.ndarray):
                        if isinstance(axs[0],np.ndarray):
                                ax_first = axs[0,j]
                        else:
                            ax_first = axs[0]
                    else:
                        ax_first = axs

                    # try to plot stimuli
                    if plot_stim:
                        stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
                        if stim_func in globals():

                            func = globals()[stim_func]
                            
                            ## retrive parameters 
                            cell = cells[0]
                            sf = cell.rec.sf
                            pad_l = int(cell.params["pre_trial"]*cell.rec.sf)

                            stim_len = len(resp_all[0])

                            func(ax_first, pad_l=pad_l, stim_len=int(stim_len), fr=sf)

                            ax_first.set_xlim(0,len(resp_all[0]))
                            ax_first.spines[['bottom', 'left', 'right', 'top']].set_visible(False)
                            ax_first.grid('dashed',linewidth = 1.5, alpha = 0.25)
                            ax_first.set_xticklabels([])
                            ax_first.set_yticklabels([])

                        else:
                            ax_first.axis("off")
                    
                    ax_first.set_title(stim, fontsize=15)

        if cbar:
            cb = fig.colorbar(ax.collections[0], cax=cbar_ax, orientation='horizontal')
            cb.set_label(label=cb_label, size=18)
            cb.ax.tick_params(labelsize=15)

        # invert trials dict
        inverted_trials_dict = {v:k for k,v in cell.sync.trials_names.items()}   

        if len(all_trials)>1:

            for i,(ax, trial) in enumerate(zip(axs[add_row:], all_trials)):
                if len(stims)>1:
                    ax[0].set_ylabel(trial,fontsize=25, color=TRIALS_C[inverted_trials_dict[trial]])
                else:
                    ax.set_ylabel(trial,fontsize=25, color=TRIALS_C[inverted_trials_dict[trial]])
            
        else:
            if isinstance(ax,np.ndarray):
                ax[0].set_ylabel(trial,fontsize=25)
            else:
                ax.set_ylabel(trial,fontsize=25)          
    
    # plot full traces
    else:

        fig, ax = plt.subplots(1,1,figsize=(10,10))

        resp_all = [eval("cell."+rtype) for cell in cells]

        resp_all = check_len_consistency(resp_all)

        # convert to array
        resp_all = np.array(resp_all)

        if normalize == "lin":

            resp_all = lin_norm(resp_all)

        elif normalize == "z":

            resp_all = z_norm(resp_all, True)
            
        # sort matrix according to quality index
        qis = [cell.qi for cell in cells]
        qis_sort = np.argsort(qis)
        resp_all = resp_all[qis_sort]

        # plot
        draw_heatmap(resp_all,vmin=vmin,vmax=vmax,cb_label=cb_label,ax=ax)

    fig.suptitle("Population Average - %d ROIs"%len(cells), fontsize=18)

    if save: 
        plt.savefig(r"%s/%s.png"%(save_path,name), bbox_inches="tight")
    # plt.close(fig)

    return fig

def plot_clusters(
    data,
    labels,
    markers=None,
    xlabel='',
    ylabel='',
    l1loc='upper right',
    l2loc='upper left',
    groups_names=None,
    grid=True,
    save=None):
    
    """
    Plot scatterplot of data. 
    Each datapoint will be color coded according to label array, and the marker will be
    assign accoding to the marker_criteria.

    - data: Array-like
        datapoints to be plotted. Only first 2 dimensions will be plotted
    - labels: Array-like
        labels array specifying the clusters
    - markers: Array-like
        markers array specifying same values for datapoints you want to draw using the
        same marker.
    - groups_names: list of str
        label prefix for the groups specifyed by the markers. only used if markers is passed
    - algo: str
        name of the embedding algorithm
    """

    clist = np.array(POPS_C)
    # allmarkers = list(Line2D.markers.items())[2:]
    # random.shuffle(allmarkers)
    # random.shuffle(clist)

    singlemarker = False

    if markers==None:

        markers = np.zeros(len(data),int).tolist()
        singlemarker = True

    Xax = data[:, 0]
    Yax = data[:, 1]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("white")

    scatters= []
    for m in np.unique(markers):

        marker = str(MARKERS[m][0])
        edgecolor = np.array(C_rand[m])*0.6
        ix = np.where(markers==m)[0]

        s = ax.scatter(
            Xax[ix], Yax[ix], 
            # edgecolors=clist[labels[ix]],
            facecolors=clist[labels[ix]]*0.7,
            edgecolor='k',
            linewidth=0.6,
            s=50, 
            marker=marker,
            alpha=0.3,
        )

        scatters.append(s)

    p = []
    l = []
    legends = []
    for i,ix in enumerate(np.unique(labels)):

        c = clist[ix]*0.7
        p.append(patches.Rectangle((0,0),1,1,fc=c))
        l.append("POP %d: %d ROIs"%(i,len(labels[labels==ix])))

    legend1 = ax.legend(p,
                        l,
                        # loc=l1loc,
                        bbox_to_anchor=(1.5, 1),
                        bbox_transform=plt.gca().transAxes
                        )
    ax.add_artist(legend1)
    legends.append(legend1)
    
    if not singlemarker:
        import matplotlib.lines as mlines

        handles = []
        for m,j in enumerate(np.unique(markers)):

            marker = str(MARKERS[m][0])
            h = mlines.Line2D([], [], marker=marker, linestyle='None',
                            markersize=10, label='%s'%(str(groups_names[j])))
            handles.append(h)
        
        legend2 = ax.legend([plt.plot([],[],marker=MARKERS[m][0],
                            color='k',#C_rand[m]*0.6, 
                            ls="none")[0] for m in np.unique(markers)],
                            ['%s'%(groups_names[i]) for i in np.unique(markers)],
                            bbox_to_anchor=(1.5, 0),
                            bbox_transform=plt.gca().transAxes
                            )
        ax.add_artist(legend2)
        legends.append(legend2)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title("%d ROIs"%(len(Xax)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks

    if grid:
        plt.grid('dashed',linewidth = 1.5, alpha = 0.25)
    
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    if save!=None:

        plt.savefig(save, bbox_extra_artists=legends, bbox_inches='tight')

    fig.subplots_adjust(right=0.75)

    return fig

def plot_receptive_fields(
        cells,
        texture_dim:tuple,
        pre_trial:int,
        avg=True,
        heatmap=False,
        rtype='norm',
        dataBehav=None,
        color=(0,0,0),
        freq=0.5,
        sr = 15.5,
        save_path=None):

    """
    Plot sparse noise responses for each cell, overlying ON and OFF responses
    for each grid location.
    - cells: list of C2p
        cells to be plotted
    - texture_dim: tuple
        tuple specifying the dimension of the texture matrix used for sparse noise stim
    - pre_trials: int
        number of frames before the trial onset included when extracting each trial response
    - dataBehav: dict
        a pair of {key:value} from a dataBehav_analyzed attribute of a rec obj
    - freq: int
        frequency of the sparse noise stim
    - sr: int
        sample rate of the recording
    - save_path: str
        path wehre to save the plots
    """

    if not avg:

        for cell in cells:

            fig, axs = plt.subplots(texture_dim[0],texture_dim[1],sharex=False,sharey=False)

            if 'sparse_noise' not in cell.analyzed_trials:

                warnings.warn("No Sparse Noise stim found for cell %s !"%cell.id, RuntimeWarning)
                continue

            ymax = 0
            ymin = 0
            for trial in cell.analyzed_trials['sparse_noise']:

                row = int(trial.split(sep='_')[0])
                col = int(trial.split(sep='_')[1])

                draw_singleStim(axs[row,col], cells, 'sparse_noise', trial, stim_window=False,color=color)
                
                axs[row,col].set_xlabel('')
                axs[row,col].set_ylabel('')
                axs[row,col].tick_params(axis=u'both', which=u'both',length=0)

                axs[row,col].set_xticks([0,1,2])
                axs[row,col].set_xticks([0,1,2])

            # plt.setp(axs, ylim=(ymin+(ymin/5),ymax+(ymax/5)))
            plt.subplots_adjust(wspace=0.04)
            plt.subplots_adjust(hspace=0.04)

            if save_path != None:
                plt.savefig(r'%s/%s.png'%(save_path,cell.id))

            # plt.close(fig)

    else:

        fig, axs = plt.subplots(texture_dim[0],texture_dim[1],figsize=(texture_dim[1]*2,texture_dim[0]*2),sharex=True,sharey=True)

        fig.suptitle('population response [df/f] - %d ROIs'%len(cells), fontsize=15)

        responses = np.zeros((texture_dim[0],texture_dim[1],len(cells)))
        trials = cells[0].analyzed_trials['sparse_noise']

        # vmax = np.max([cell.analyzed_trials['sparse_noise'][trial]["%s_avg"%rtype].max() for  cell in cells for trial in trials])
        # vmin = np.min([cell.analyzed_trials['sparse_noise'][trial]["%s_avg"%rtype].min() for  cell in cells for trial in trials])


        for trial in trials:

            row = int(trial.split(sep='_')[0])
            col = int(trial.split(sep='_')[1])

            if heatmap:
                # extract data from each cell
                resp_all = []
                for cell in cells:

                    r = cell.analyzed_trials['sparse_noise'][trial]["%s_avg"%rtype]
                    resp_all.append(r)

                resp_all = check_len_consistency(resp_all)
                # convert to array
                resp_all = np.array(resp_all)

                # normalize
                resp_all = z_norm(resp_all)

                # sort matrix according to quality index
                qis = [cell.qi for cell in cells]
                qis_sort = np.argsort(qis)
                resp_all = resp_all[np.flip(qis_sort)]

                draw_heatmap(resp_all,cbar=False,ax=axs[row,col])   
                axs[row,col].axvline(cell.analyzed_trials['sparse_noise'][trial]['window'][0],0,color='r',linestyle='--',alpha=0.8) 
                axs[row,col].axvline(cell.analyzed_trials['sparse_noise'][trial]['window'][1],0,color='r',linestyle='--',alpha=0.8) 

            else:
                draw_singleStim(axs[row,col], cells, 'sparse_noise', trial, stim_window=False, color=color)
           
            axs[row,col].set_xlabel('')
            axs[row,col].set_ylabel('')
            axs[row,col].tick_params(axis=u'both', which=u'both',length=0)

            axs[row,col].set_xticks([0,1,2])
            axs[row,col].set_xticks([0,1,2])
        
            plt.subplots_adjust(wspace=0.04)
            plt.subplots_adjust(hspace=0.04)
        
        if save_path != None:
            plt.savefig(r'%s.png'%(save_path), transparent=True)

        # plt.close(fig)

        # plot also dataBehav if desired
        if dataBehav != None:

            fig, axs = plt.subplots(texture_dim[0],texture_dim[1],figsize=(texture_dim[1]*2,texture_dim[0]*2),sharex=True,sharey=True)

            title = ''
            for db in dataBehav.keys(): title+='%s  '%(str(db))
            
            fig.suptitle(title+'   [zscore]', fontsize=15)

            for (name,data),color in zip(dataBehav.items(),[[0.,0.,0.],[0.8,0.1,0.1]]):

                responses = np.zeros((texture_dim[0],texture_dim[1],len(cells)))
                trials = data['sparse_noise']
                
                for trial in trials:

                    row = int(trial.split(sep='_')[0])
                    col = int(trial.split(sep='_')[1])
            
                    draw_singleStim_dataBehav(axs[row,col], {name:data}, 'sparse_noise', trial, rtype='norm', color=np.array(color), stim_window=False)
                    axs[row,col].set_xlabel('')
                    axs[row,col].set_ylabel('')
                    axs[row,col].tick_params(axis=u'both', which=u'both',length=0)

                    axs[row,col].set_xticks([0,1,2])
                    axs[row,col].set_xticks([0,1,2])
                
                    plt.subplots_adjust(wspace=0.04)
                    plt.subplots_adjust(hspace=0.04)
            
            if save_path != None:
                    plt.savefig('%s_%s.png'%(save_path,title), transparent=True)

            # plt.close(fig)
             
def plot_histogram(
    values:dict, 
    xlabel='',
    binw=None,
    kde=False,
    alpha=0.6,
    colors=sns.color_palette('pastel'),
    grid=False,
    save=None):

    """
    Plot distribution of the data contained in values array.
    If values is a matrix, histograms will be computed on the axis 1 (coulumns).

    - values: dict
        dict containing stim:[rmis] items that will be used for plotting the histograms.


    """
    import pandas as pd

    df = pd.DataFrame(values)

    sns.displot(data=df, binwidth=binw, palette=colors[:len(values)], edgecolor=(0,0,0), stat='percent',
                rug=False, height=4, kde=kde, legend=True, alpha=alpha) 

    plt.axvline(0,0,1,c=(0.1,0.1,0.1),linewidth=1.5,ls='-.', alpha=0.8)
    
    plt.xlim(-1,1)
    plt.xlabel(xlabel)
    plt.gca().spines[['right','top','left','bottom']].set_visible(False)
    if grid:
        plt.grid('dashed',linewidth = 1.5, alpha = 0.15,)

    if save != None:
        plt.savefig(save, bbox_inches='tight')
        
    # plt.show()

    return plt.gca()
