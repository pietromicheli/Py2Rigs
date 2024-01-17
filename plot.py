import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib import cm,patches
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


cmap = cm.get_cmap('rainbow')
rng = np.random.default_rng(seed=999)
index = rng.random(100)
TRIALS_C = [cmap(i) for i in index]

        # POPS_C = list(colors.TABLEAU_COLORS.keys())
POPS_C  = [np.array(c) for c in sns.color_palette('pastel')]

### DRAWING FUNCTIONS ###

def draw_singleStim(
    ax,
    cells,
    stim,
    trials,
    rtype="norm",
    ylabel="",
    stim_window = True,
    legend = False,
    func_ = None
):

    """
    Plot average response calculated across all the specified cells .

    - cells: 
        cell to plot. if a list of C2p object is passed, plot the mean. 
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

            ax.axvspan(on_frame, off_frame, color="k", alpha=0.7)
            # only draw stim window for first trial
            stim_window = False

            ax.set_xticks(
                [0, on_frame, on_frame+1, off_frame, int(len(r) / sf)]
            )

            ax.set_xticklabels(
                [round(-on_frame,1), 0, 1, round(off_frame-on_frame,1), round(len(r) / sf,1)]
            )

        ax.plot(x, r, c=TRIALS_C[inverted_trials_dict[trial]], linewidth=0.8, alpha=0.6, label=trial)

        if isinstance(error,np.ndarray):

            ax.fill_between(
                x, r + error, r - error, color=np.array(TRIALS_C[inverted_trials_dict[trial]])*0.5, alpha=0.3
            )

        title += " %s"%trial

        if ylabel==None:
        
            if rtype == "norm":

                ax.set_ylabel("\u0394F/F",fontsize=18)
            
            if rtype == "spks":
                
                ax.set_ylabel("OASIS spikes",fontsize=18)

            if rtype == "zspks":
                
                ax.set_ylabel("OASIS spikes (z-score)",fontsize=18)

        ax.set_ylabel(ylabel,fontsize=18)

        ax.set_xlabel("time [s]",fontsize=18)

        if legend:

            ax.legend()

        # ax.set_title(title,fontsize=18)

    ax.spines[['right','top','left','bottom']].set_visible(False)
    ax.grid('dashed',linewidth = 1.5, alpha = 0.25)

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

        ax.set_ylabel(ylabel, fontsize=18)

    for stim in stims:

        for trial_type in list(sync.sync_ds[stim].keys())[:-1]:

            c = TRIALS_C[inverted_trials_dict[trial_type]]

            for i,trial in enumerate(sync.sync_ds[stim][trial_type]["trials"]):

                ax.axvspan(int((trial[0]-offset)/sf), int((trial[1]-offset)/sf),
                                color=c, alpha=0.2, label="_"*i+"%s -%s"%(stim,trial_type))
                
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

        ax.set_ylabel(ylabel, fontsize=18)

    return ax

def draw_heatmap(
        matrix, 
        vmin, 
        vmax,
        cbar=False,
        cb_label="", 
        ax=None):

    """
    Base function for drawing an hetmap from input matrix.
    """

    m = np.array(matrix)

    map = sns.color_palette("icefire", as_cmap=True)


    ax = sns.heatmap(
        m,
        vmin,vmax,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar=cbar,
        cmap=map,
        cbar_kws=dict(
        use_gridspec=False, location="bottom", pad=0.05, label=cb_label, aspect=40,
        ),
    )

    ax.figure.axes[-1].xaxis.label.set_size(15)

    return ax

### PLOTTING FUNCTIONS ###

def plot_multipleStim(
    cells: list,
    stim_dict,
    average=True,
    rtype="norm",
    func_=None,
    full="norm",
    stim_window=True,
    ylabel='',
    qi_threshold=0,
    plot_stim =True,
    order_stims=False,
    group_trials=True,
    share_x=False,
    share_y=False,
    legend=False,
    save=False,
    save_path="",
    save_suffix="",
):

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
    add_row = 0
    add_full = 0
    add_stim = 0

    if full != None:

        add_full = 1

    for stim in stims:

        stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
        if plot_stim and stim_func in globals():

            add_stim = 1

    if add_stim == 0:
        plot_stim = False

    add_row = add_full+add_stim

    # main loop
    for c in cells:

        if group_trials:
            figsize = (9*n_stims, 3*(2+add_row))
            nrows = (1+add_row)
        else:
            figsize = (9*n_stims, 8*n_trials+add_row)
            nrows=(n_trials+add_row)

        height_ratios = [1]*nrows

        if plot_stim:
            height_ratios[0] /= 2
                       
        fig, axs = plt.subplots(nrows=nrows, ncols=n_stims, figsize=figsize,
                                gridspec_kw={'height_ratios': height_ratios})

        if average:

            fig.suptitle("Population Average - %d ROIs"%len(c),fontsize=18)

            # retrive syncs
            sync = c[0].sync
            # spikes?
            spks = (c[0].rec.dtype=='spikes')

        else:

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

        for j,stim in enumerate(stims):

            # y_max = 0
            # y_min = 0

            trials_ = sorted(stim_dict[stim])

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

                _, ymax, ymin = draw_singleStim(axs_, c, stim, trials_,rtype, ylabel=ylabel, 
                                                stim_window=stim_window,func_=func_, legend=legend)

                if ymax > y_max: y_max = ymax

                if ymin < y_min: y_min = ymin

                # if j>0: axs_.set_ylabel("")

                if full:

                    axs_.set_xlabel("")

            else:
                
                axs_T = axs.T

                if len(stims)>1:

                    axs_T = axs_T[j]

                trials_ = sorted(stim_dict[stim])

                for i,trial in enumerate(trials_):

                    if plot_stim:

                        i +=1

                    ## make sure to use only trials which exist for a specific stim
                    if trial in sync.sync_ds[stim]:

                        _, ymax, ymin = draw_singleStim(axs_T[i], c, stim, trial,rtype, ylabel=ylabel+' [%s]'%rtype,
                                                        stim_window=stim_window, func_=func_, legend=legend)

                        if ymax > y_max: y_max = ymax

                        if ymin < y_min: y_min = ymin

                        if i>0: axs_T[i].set_title("")

                    else:
                        
                        axs_T[i].axis("off")
                        axs_T[i].set_title("")

                        if i>0: axs_T[i].set_title("")

            if full != None:
                
                if axs.ndim==1:

                    ax_full = axs[-1]

                else:

                    ax_full = axs[-1,j]

                if full == 'norm' or full == 'raw': 
                    ylabel_full = ylabel+' [%s]'%full

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

                if stim_func in globals():

                    func = globals()[stim_func]
                    ## retrive parameters 
                    cell = c
                    if isinstance(c, list):
                        cell = c[0]

                    sf = cell.rec.sf
                    pad_l = int(cell.params["pre_trial"]*cell.rec.sf)

                    stim_len = 0
                    for ax in axs_T[1:]:

                        if ax.lines:

                            stim_len = ax.lines[0].get_xdata().size
                            break                

                    func(axs_T[0], pad_l=pad_l, stim_len=int(stim_len), fr=sf)

                    axs_T[0].spines[['right','top','left','bottom']].set_visible(False)
                    axs_T[0].grid('dashed',linewidth = 1.5, alpha = 0.25)
                    axs_T[0].set_xticklabels([])
                    axs_T[0].set_yticklabels([])

                else: axs_T[0].axis("off")
            
                axs_T[0].set_title(stim, fontsize=15)  
        
        if plot_stim: axs= axs [1:]
        
        if full != None: axs = axs[:-1]

        # for ax in axs.flatten():

        #     ax.set_ylim(y_min-(abs(y_min/10)), y_max+(abs(y_max/10)))

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
    
def plot_FOV(
        rec,
        cells_ids,
        save_path="FOV.png", 
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

    mean_img = rec.ops.item()[img]

    img_rgb = ((np.stack([mean_img,mean_img,mean_img],axis=2)-np.min(mean_img))/np.max((mean_img-np.min(mean_img))))
    k = 0.22/np.mean(img_rgb)
    img_rgb = img_rgb*k

    plt.figure(figsize=(10,7))

    all_labels = []

    if cells_ids != None:
        # for i,pop in enumerate(cells_ids):
        for idx in cells_ids:

            # c = POPS_C[i]
            pop = rec.cells[idx].label
            all_labels.append(pop)
            c = POPS_C[pop]

            # for idx in pop:

            # extract and color CELLs pixels

            ypix = rec.stat[idx]['ypix']

            xpix = rec.stat[idx]['xpix']

            for x,y in zip(xpix,ypix):

                img_rgb[(y),(x)] = colors.to_rgb(c)

        for pop in set(all_labels):

            plt.plot(0,0,c=POPS_C[pop],label='POP_#%d'%pop)
            leg = plt.legend(loc="upper right", bbox_to_anchor=(1.16, 1.0),facecolor='white')
            leg.get_frame().set_linewidth(0.0)

    plt.imshow(img_rgb, aspect='auto')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches="tight")
    
def plot_heatmaps(
    cells,
    stim_dict,
    rtype="norm",
    full=None,
    vmin=None,
    vmax=None,
    normalize=None,
    save=True,
    save_path="",
    name="",
    cb_label="",
):

    """
    Plot heatmap for all the cells.

    - cells: list
        list of C2p objects
    - stim_dict: dict
        dict containing stim:[trials] items specyfying what to plot
    - stims: list
        list of stimuli names to plot
    - trials: list
        list of trials names to plot
    -rtype: str
        can be "norm" or "zspks"
    - full: str
        can be Fraw,norm,spks or zspks. if specified, stims,trials andrtype
        argoument will be ignored
    """

        
    # plot averages
    if full == None:

        stims = list(stim_dict.keys())

        all_trials = []

        for stim in stim_dict:
            all_trials = all_trials+stim_dict[stim]

        all_trials = set(sorted(all_trials))

        add_row = 0
        for stim in stims:

            stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
            if stim_func in globals():
                add_row=1

        nrows = len(all_trials)+add_row
        height_ratios = [1]*nrows

        if add_row:
            height_ratios[0] /= 2
                       
        fig, axs = plt.subplots(nrows=nrows, ncols=len(stims),figsize=(25,20), 
                                gridspec_kw={'height_ratios': height_ratios})

        cbar_ax = fig.add_axes([.3, .05, .4, .015])
        cbar = False

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

                    if normalize == "linear":

                        resp_all = lin_norm(resp_all)

                    elif normalize == "zscore":

                        resp_all = z_norm(resp_all, True)

                    # sort matrix according to quality index
                    qis = [cell.qi for cell in cells]
                    qis_sort = np.argsort(qis)
                    resp_all = resp_all[np.flip(qis_sort)]

                    # plot
                    # if i==len(all_trials):
                    #     cbar=True

                    draw_heatmap(resp_all,vmin=vmin,vmax=vmax,cb_label=cb_label,cbar=cbar,ax=ax)    

                    # try to plot stimuli
                    stim_func = "%s_stim"%(stim.split(sep='-')[0]) 
                    if stim_func in globals():

                        func = globals()[stim_func]
                        
                        ## retrive parameters 
                        cell = cells[0]
                        sf = cell.rec.sf
                        pad_l = int(cell.params["pre_trial"]*cell.rec.sf)

                        stim_len = len(resp_all[0])

                        func(axs[0,j], pad_l=pad_l, stim_len=int(stim_len), fr=sf)

                        axs[0,j].set_xlim(0,len(resp_all[0]))
                        axs[0,j].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
                        axs[0,j].grid('dashed',linewidth = 1.5, alpha = 0.25)
                        axs[0,j].set_xticklabels([])
                        axs[0,j].set_yticklabels([])

                    else:
                        axs[0,j].axis("off")
                    
                    axs[0,j].set_title(stim, fontsize=15)

        cb = fig.colorbar(ax.collections[0], cax=cbar_ax, orientation='horizontal')
        cb.set_label(label=cb_label, size=18)
        cb.ax.tick_params(labelsize=15)

        if len(all_trials)>1:

            for ax, trial in zip(axs[add_row:], all_trials):
                ax[0].set_ylabel(trial,fontsize=18)

        else:
            ax[1].set_ylabel(trial,fontsize=18)
        

    # plot full traces
    else:

        fig, ax = plt.subplots(1,1,figsize=(20,20))

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

def plot_clusters(
    data,
    labels,
    markers=None,
    xlabel='',
    ylabel='',
    l1loc='upper right',
    l2loc='upper left',
    groups_names=None,
    save=None

):
    
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
    allmarkers = list(Line2D.markers.items())[2:] 
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

        marker = str(allmarkers[m][0])
        ix = np.where(markers==m)[0]

        s = ax.scatter(
            Xax[ix], Yax[ix], 
            # edgecolors=clist[labels[ix]],
            facecolors=clist[labels[ix]]*0.7,
            s=50, 
            marker=marker,
            alpha=0.2,
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
                        bbox_to_anchor=(1.35, 1.0),
                        bbox_transform=plt.gca().transAxes
                        )
    ax.add_artist(legend1)
    legends.append(legend1)
    
    if not singlemarker:
        import matplotlib.lines as mlines

        handles = []
        for m,j in enumerate(np.unique(markers)):

            marker = str(allmarkers[m][0])
            h = mlines.Line2D([], [], marker=marker, linestyle='None',
                          markersize=10, label='%s'%(str(groups_names[j])))
            handles.append(h)
        
        legend2 = ax.legend([plt.plot([],[],marker=allmarkers[m][0],color='k', ls="none")[0] for m in np.unique(markers)],
                            ['%s'%(groups_names[i]) for i in np.unique(markers)],
                            bbox_to_anchor=(1.35, 0),
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

    plt.grid('dashed',linewidth = 1.5, alpha = 0.25)

    if save!=None:

        plt.savefig(save, bbox_extra_artists=legends, bbox_inches='tight')

def plot_sparse_noise(
        cells,
        texture_dim:tuple,
        pre_trial:int,
        freq=4,
        sr = 15.5,
        save_path=''
):

    """
    Plot sparse noise responses for each cell, overlying ON and OFF responses
    for each grid location.
    - cells: list of C2p
        cells to be plotted
    - texture_dim: tuple
        tuple specifying the dimension of the texture matrix used for sparse noise stim
    - pre_trials: int
        number of frames before the trial onset included when extracting each trial response
    - freq: int
        frequency of the sparse noise stim
    - sr: int
        sample rate of the recording
    - save_path: str
        path ehre to save the plot
    """

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
            rtype = trial.split(sep='_')[2]

            r = cell.analyzed_trials['sparse_noise'][trial]['avg_norm']
            std = cell.analyzed_trials['sparse_noise'][trial]['std_norm']
            x = np.arange(len(r))

            if rtype=='on':
                c = 'r'
            elif rtype=='off':
                c = 'k'

            axs[row,col].plot(r,linewidth=0.7,alpha=0.7,c=c,label=str(cell.analyzed_trials['sparse_noise'][trial]["QI"]<0.05))
            axs[row,col].fill_between(x,r+std,r-std,alpha=0.1,color=c)
            axs[row,col].axvspan(pre_trial,(pre_trial+int(sr/freq)),alpha=0.1,color='y')
            axs[row,col].legend(fontsize=9)
            if (row+col)==0:
                axs[row,col].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

            else:
                axs[row,col].set_xticks([])
                axs[row,col].set_yticks([])

            if (r+std).max()>ymax: ymax = (r+std).max()
            if (r-std).min()<ymin: ymin = (r-std).min()


        plt.setp(axs, ylim=(ymin+(ymin/5),ymax+(ymax/5)))
        plt.subplots_adjust(wspace=0.04)
        plt.subplots_adjust(hspace=0.04)

        plt.savefig(r'%s/%s.png'%(save_path,cell.id))
        plt.close(fig)

def plot_histogram(
    values:dict, 
    control_values=None,
    bins=None,
    colors=sns.color_palette('pastel'),
    save_name="hist.png"):

    """
    Plot distribution of the data contained in values array.
    If values is a matrix, histograms will be computed on the axis 1 (coulumns).

    - values: dict
        dict containing stim:[rmis] items that will be used for plotting the histograms.
    - control values: dict
        dict containing stim:[rmis] items that will be used for plotting the control histograms.

    """

    fig, axs = plt.subplots(figsize=(10,10),ncols=1)

    if not isinstance(axs, np.ndarray):

        axs = np.array([axs])

    for i, stim in enumerate(values):

        if bins == None:
            q25, q75 = np.percentile(values[stim], [25, 75])
            bin_width = 2 * (q75 - q25) * len(values[stim]) ** (-1/3)
            bins = round((max(values[stim]) - min(values[stim])) / bin_width)

        # axs[0].set_title(stim, fontsize=12)
        norm_values = values[stim]/(values[stim].max())
        # y, x , _ = axs[0].hist(norm_values,bins=bins,alpha=0.3,edgecolor='black',color=colors[i],
        #                        label='condition %d'%i, density=True)
        sns.distplot(values[stim], bins, True, color=colors[i].tolist(), label='condition %d'%i)   
        
        if control_values != None:

            q25, q75 = np.percentile(control_values[stim], [25, 75])
            bin_width = 2 * (q75 - q25) * len(control_values[stim]) ** (-1/3)
            bins = round((max(control_values[stim]) - min(control_values[stim])) / bin_width)

            axs[0].hist(control_values[stim],bins=bins,alpha=0.5,edgecolor='black',color='k')

        # max_y = np.max(y)
        # mean_y = np.median(y)
        # max_y_i = np.argmax(y)
        # mean_y_i = (np.abs(y - mean_y)).argmin()
        # max_x = (x[max_y_i]+x[(max_y_i+1)])/2
        # mean_x =(x[mean_y_i]+x[(mean_y_i+1)])/2

    # axs[i].axvline(mean_x,0,(max_y+10),c='r',linewidth=1.7)
    axs[0].axvline(0,0,1,c='k',linewidth=1,ls='-.')
    
    axs[0].set_xlim(-1,1)
    axs[0].set_xlabel('RMI')
    axs[0].set_ylabel('normalized counts')

    axs[0].spines[['right','top','left','bottom']].set_visible(False)
    plt.grid('dashed',linewidth = 1.5, alpha = 0.25)
    plt.legend()

    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# def plot_clusters(
#     data,
#     labels,
#     markers=None,
#     xlabel='',
#     ylabel='',
#     l1loc='upper right',
#     l2loc='upper left',
#     groups_name='Group',
#     save=None

# ):
    
#     """
#     Plot scatterplot of data. 
#     Each datapoint will be color coded according to label array, and the marker will be
#     assign accoding to the marker_criteria.

#     - data: Array-like
#         datapoints to be plotted. Only first 2 dimensions will be plotted
#     - labels: Array-like
#         labels array specifying the clusters
#     - markers: Array-like
#         markers array specifying same values for datapoints you want to draw using the
#         same marker.
#     - groups_name: str
#     label prefix for the groups specifyed by the markers. only used if markers is passed
#     - algo: str
#         name of the embedding algorithm
#     """

#     clist = np.array(POPS_C)
#     allmarkers = list(Line2D.markers.items())[2:] 
#     # random.shuffle(allmarkers)
#     # random.shuffle(clist)

#     singlemarker = False

#     if markers==None:

#         markers = np.zeros(len(data),int).tolist()
#         singlemarker = True

#     Xax = data[:, 0]
#     Yax = data[:, 1]

#     fig = plt.figure(figsize=(7, 5))
#     ax = fig.add_subplot(111)

#     fig.patch.set_facecolor("white")

#     for m in np.unique(markers):

#         marker = allmarkers[m][0]
#         ix = np.where(markers==m)[0]

#         s = ax.scatter(
#             Xax[ix], Yax[ix], 
#             # edgecolors=clist[labels[ix]],
#             facecolors=clist[labels[ix]]*0.7,
#             s=50, 
#             marker=marker,
#             alpha=0.2,
#         )

#     p = []
#     l = []
#     for i,ix in enumerate(np.unique(labels)):

#         c = clist[ix]*0.7
#         p.append(patches.Rectangle((0,0),1,1,fc=c))
#         l.append("POP %d: %d ROIs"%(i,len(labels[labels==ix])))

#     legend1 = ax.legend(p,
#                         l,
#                         # loc=l1loc,
#                         bbox_to_anchor=(1.35, 1.0),
#                         bbox_transform=plt.gca().transAxes
#                         )
#     ax.add_artist(legend1)
    
#     if not singlemarker:
#         import matplotlib.lines as mlines

#         handles = []
#         for m in np.unique(markers):
#             h = mlines.Line2D([], [], marker=m, linestyle='None',
#                           markersize=10, label='%s %d'%(groups_name,i))
#             handles.append(h)
        
#         legend2 = ax.legend([plt.plot([],[],marker=allmarkers[m][0],color='k', ls="none")[0] for m in np.unique(markers)],
#                             ['%s %d'%(groups_name,i) for i in np.unique(markers)],
#                             bbox_to_anchor=(1.35, 0),
#                             bbox_transform=plt.gca().transAxes
#                             )
#         ax.add_artist(legend2)
#         legends = (legend1,legend2)

#     else: legends=(legend1)
    

#     ax.set_xlabel(xlabel, fontsize=18)
#     ax.set_ylabel(ylabel, fontsize=18)
#     ax.set_title("%d CELLs"%(len(Xax)))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)

#     plt.grid('dashed',linewidth = 1.5, alpha = 0.25)

#     if save!=None:

#         plt.savefig(save, bbox_extra_artists=(legends,), bbox_inches='tight')
