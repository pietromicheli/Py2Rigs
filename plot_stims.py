import numpy as np
from scipy.special import sinc
from scipy.signal import chirp
from matplotlib.patches import Rectangle

def general_stim(
        frames_tot,
        stim_l,
        ax=None,
        pad_l = 40):
    

    ax.axvspan(0,pad_l, 0,0.5, color='w')
    ax.axvspan(pad_l,pad_l+stim_l, 0,0.5,  color='k')
    ax.axvspan(pad_l+stim_l,frames_tot, 0,0.5, color='w')

    return ax


def chirp_stim(
        frames_tot,
        ax=None,
        pad_l = 40,
        freq_cr = 0.55,
        fr = 15.5,
        stim_l=None):

    pre_trial = np.ones(pad_l)*0.5
    chunk_OFF1 = np.zeros(int(2*fr))
    chunk_ON = np.ones(int(3*fr))
    chunk_OFF2 = np.zeros(int(3*fr))
    BG = np.ones(int(2*fr))*0.5
    t_fm = np.linspace(0,20,int(8*fr))
    FM = (chirp(t_fm,0.2,40,1)+1)/2
    ttot = 12/freq_cr
    t_am = np.linspace(0,ttot,int(ttot*fr))
    # a_am = np.linspace(0.01,0.5,int(21*fr))
    scale = t_am/ttot
    AM = (np.sin(t_am*2*np.pi*freq_cr))/2*scale+0.5
    
    stim_conc = np.concatenate([pre_trial,chunk_OFF1,chunk_ON,chunk_OFF2,chunk_ON*0.5,
                                FM,BG,AM,BG,chunk_OFF1,chunk_ON,chunk_OFF1,chunk_ON,chunk_OFF1])
    
    pad_right = np.ones((frames_tot-len(stim_conc)))*0.5
    stim_conc = np.concatenate([stim_conc, pad_right])

    if ax != None:
        ax.plot(stim_conc,c='k')
        green_x = len(stim_conc)-len(pad_right)-int(10*fr)
        blue_x = len(stim_conc)-len(pad_right)-int(5*fr)

        ax.axvspan(green_x,green_x+len(chunk_ON), 0, 1, color='g',alpha=0.3)
        ax.axvspan(blue_x,blue_x+len(chunk_ON), 0, 1, color='b',alpha=0.3)
        
        ax.set_xticks(ax.get_xticks()[1:-1], (ax.get_xticks()[1:-1]/fr).astype(int))
        ax.set_ylabel("brightness", fontsize=18)

    return ax,stim_conc

def chirp_LED_stim(
        ax=None,
        pre_trial = 40,
        frames_tot = None,
        freq_cr = 0.55,
        fr = 15.5):

    pre_trialeft = np.ones(pre_trial)*0.5
    chunk_OFF1 = np.zeros(int(2*fr))
    chunk_ON = np.ones(int(3*fr))
    chunk_OFF2 = np.zeros(int(3*fr))
    BG = np.ones(int(2*fr))*0.5
    t_fm = np.linspace(0,8,int(8*fr))
    FM = (chirp(t_fm,f0=0.25,f1=1,t1=8,phi=-90)+1)/2
    frames_tot = 12/freq_cr
    t_am = np.linspace(0,frames_tot,int(frames_tot*fr))
    # a_am = np.linspace(0.01,0.5,int(21*fr))
    scale = t_am/frames_tot
    AM = (np.sin(t_am*2*np.pi*freq_cr))/2*scale+0.5
    
    stim_conc = np.concatenate([pre_trialeft,chunk_OFF1,chunk_ON,chunk_OFF2,chunk_ON*0.5,
                                FM,BG,AM,BG])
    
    pad_right = np.ones((frames_tot-len(stim_conc)))*0.5
    stim_conc = np.concatenate([stim_conc, pad_right])

    if ax != None:
        ax.plot(stim_conc,c='k')
        
        ax.set_xticks(ax.get_xticks()[1:-1], (ax.get_xticks()[1:-1]/fr).astype(int))
        ax.set_ylabel("brightness", fontsize=18)

    return ax,stim_conc

def full_field_stim(
        frames_tot,
        stim_l,
        ax=None,
        pad_l = 40,
        fr = 15.5):


    pre_trial = np.zeros(pad_l)+0.25 # 50% contrast
    chunk_1 = np.ones(stim_l)*0.55

    stim_conc = np.concatenate([pre_trial,chunk_1])
    pad_right = np.zeros((frames_tot-len(stim_conc)))+0.25

    stim_conc = np.concatenate([stim_conc, pad_right])  

    stim_start = len(stim_conc)-len(pad_right)-stim_l
    stim_end = len(stim_conc)-len(pad_right)

    if ax != None:
        ax.plot(stim_conc,c='k')
        ax.set_ylim(0,1)
        ax.axvspan(stim_start, stim_end, color='y', alpha=0.1)
        ax.set_ylabel("brightness", fontsize=13)

    return ax,stim_conc

def contrast_ramp_stim(
    frames_tot,
    ax=None,
    stim_l=None,
    pad_l = 40,
    fr = 15.5,
    freq = 0.55 #0.57
    ):

   ncycles = 12
   frames_tot = int(ncycles/freq)

   pre_trialeft = np.ones(pad_l)*0.5
   t_am = np.linspace(0,frames_tot,int(frames_tot*fr)) 
   scale = t_am/21
   AM = (np.sin(t_am*2*np.pi*freq))/2*scale+0.5
   
   stim_conc = np.concatenate([pre_trialeft,AM])
   
   pad_right = np.ones((frames_tot-len(stim_conc)))*0.5
   stim_conc = np.concatenate([stim_conc, pad_right])

   if ax!=None:
    ax.plot(stim_conc,c='k')
    ax.set_xticks(ax.get_xticks()[1:-1], (ax.get_xticks()[1:-1]/fr).astype(int))
    ax.set_ylabel("brightness", fontsize=18)

   return ax,stim_conc
