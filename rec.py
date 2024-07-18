import numpy as np
import yaml
import json
import warnings
from copy import copy
from tqdm import tqdm
from sklearn.decomposition import PCA
from umap import UMAP

from .sync import Sync
from .cell import Cell

from .loader import Dataloader
from .utils import *
from .plot import plot_clusters

warnings.simplefilter('always')

class Rec:

    """
    A recording master class
    """

    def __init__(self, loader: Dataloader, id=0):

        """
        Create a Rec object using the data found in the Dataloader input object

        """

        if not isinstance(loader.dataRaw, np.ndarray) and not loader.dataBehav:
            raise Exception("ERROR: no Data found in the Loader!")            

        self.id = id
        self.loader = loader
        self.dataRaw = loader.dataRaw
        self.dataBehav = loader.dataBehav
        self.dtype = loader.dtype
        self.precomputed = loader.precomputed
        self.sync = copy(loader.sync)
        self.group = loader.group
        self.rec_length = loader.rec_length
        self.ncells = loader.ncells
        self.cells = {}
        self.responsive = []
        self.populations = []
        self.dataBehav_analyzed = {}

        # IF THE LOADER CONTAINS PRE-EXTRACTE DATA, LOAD THEM!
        if self.precomputed == True:
            
            with h5py.File(loader.h5_path, "r") as h5file:

                print('> Building the data structure ...', end='')          
                self.responsive = h5file['responsive'][()]
                self.populations = h5file['populations'][()]
                self.params =  hdf5_to_dict(h5file['params'])

                if 'ops' in h5file.keys():
                    # load only mean image, is faster
                    self.ops =  h5file['ops']['meanImg'][()]
                if 'stat' in h5file.keys():
                    self.stat =  hdf5_to_dict(h5file['stat'])

                cells_pbar = tqdm(list(h5file['cells'].keys()))
                for id in cells_pbar:
                    
                    cell_dict = hdf5_to_dict(h5file['cells'][id])
                    cell = Cell(self, id)

                    for key,value in cell_dict.items():
                        cell.__dict__[key] = value
                    
                    self.cells |= {id:cell}
                print('OK!')

        else:
            # SET PARAMETERS
            self.params_file = generate_params_file()
            self.params = None

            # load ops and stat from s2p if available
            if 'ops' in loader.__dict__:
                self.ops = loader.ops
            if 'stat' in loader.__dict__:
                self.stat = loader.stat

            # read parameters from .yaml params file
            self.load_params()

            # sample frequency
            if self.params['binsize'] >0: self.sf = int(self.params['sf']/self.params['binsize']) # DOUBLE CHECK!! #
            else: self.sf = self.params['sf']

            # if no sync obj is found inside the loader, 
            # create a basic one, containing only one stimulus and one trial
            if self.sync == None:

                warnings.warn("No Sync found in the loader!\n", RuntimeWarning)

                frames = np.array([self.params['baseline_length']*self.sf,
                                self.rec_length - self.params['max_aftertrial']*self.sf])
                stim_dict = {'stim':[0]}

                np.save('sync_temp.npy',frames)
                with open('stim_dict_temp.json', 'w') as fp:
                    json.dump(stim_dict, fp)

                self.sync = Sync().generate_data_structure('sync_temp.npy', 
                                                        'stim_dict_temp.json')

                os.remove('sync_temp.npy')
                os.remove('stim_dict_temp.json')

            # Bin Data and Sync if required
            if self.params["binsize"] > 0:

                self._bin_data_(self.params["sf"]*self.params["binsize"])

                # if spikes, compute firing rate
                if self.loader.dtype == 'spikes':
                    self.dataRaw /= self.params["binsize"]

            # Filter data 
            if self.dataRaw.size > 0:
            
                self.dataRaw = filter(self.dataRaw, self.params['lowpass_wn'])
                
                # Normalize the raw data:

                # if data contains fluo traces, compute df/f normalization
                if self.dtype == 'fluo':

                    # subtract neuropil if present
                    if hasattr(self.loader,'Fneu'):

                        self.dataRaw -= self.loader.Fneu*self.params['neuropil_corr']

                    if self.params["baseline_indices"] is not None:
                        bs_indices = self.params["baseline_indices"]
                    else:
                        bs_indices = [0,self.sync.sync_tps[0]]
                        
                    baseline = np.mean(self.dataRaw[:,bs_indices[0]:bs_indices[1]],axis=1)
                    baseline = np.expand_dims(baseline,1)
                    self.dataNorm = (self.dataRaw - baseline) / baseline

                # if data contains spiking activity, compute z-score 
                elif self.dtype == 'spikes':
                    # resps_std = np.std(resps, axis=1)
                    # resps_zscore = ((resps.T - mean_baselines)/resps_std).T
                    self.dataNorm = z_norm(self.dataRaw)

    def load_params(self):

        """
        Read parameters from .yaml params file
        """

        with open(self.params_file, "r") as f:

            self.params = yaml.load(f, Loader=yaml.Loader)

            # update also for all the cells
            if self.cells != None:

                for cell in self.cells:

                    self.cells[cell].params = self.params

            print("> parameters loaded.")

        return self.params

    def get_responsive(self):

        """
        Get a list containing the ids of all the responsive cells
        """

        ids = []

        for cell in self.cells:

            if self.cells[cell].responsive:

                ids.append(cell)

        return ids

    def extract_all(self, keep_unresponsive=False, save_hdf5_filename:str=None):

        """
        Extract neural rsponse from cells and behavioral data if available
        """

        cells = self._extract_data_(keep_unresponsive=keep_unresponsive)
        behavior = self._extract_dataBehav_()

        if save_hdf5_filename != None:
            # save analyzed results in hdf5 file
            cells = {'cells':{cid:{key:value for key,value in c.__dict__.items() if key not in ['rec','sync','params','norm','raw']} for cid,c in self.cells.items()}}
            
            sync = {'sync': {key:value for key,value in self.sync.__dict__.items()}}
            rec = {key:value for key,value in self.__dict__.items() if key!='loader'}
            DATA = {**rec, **sync, **cells}
            
            print('\n> Saving extracted data in hdf5 file...', end='')
            save_dict_to_hdf5(DATA, save_hdf5_filename+'.hdf5')
            print('OK!')

        return self.cells

    def compute_fingerprints(
        self, 
        cells_ids=None,
        stim_trials_dict=None, 
        rtype="norm", 
        mode="full",
        normalize="zscore", 
        smooth=True):

        """
        Compute a fingerprint for each cell by concatenating the average responses
        to the specified stimuli and trials.

        - stim_trials_dict: dict
            A dict which specifies which stim and which trials to concatenate for computing
            the fingerptint.
            Should contain key-values pairs such as {stim:[t1,...,tn]}, where stim is a valid
            stim name and [t1,...,tn] is a list of valid trials for that stim.
        - rtype: str
            whether to use normalized or raw responses. Can be either 'norm' or 'raw'
        - mode: str
            can be full (concatenate the full responses to trials) or median (concatenate the media of responses to trials) 

        """

        # check if the cells have already been retrived

        if self.cells == None:
            self.get_cells()

        if stim_trials_dict == None:
            stim_trials_dict = {stim: list(self.sync.sync_ds[stim].keys())[:-1] 
                                for stim in self.sync.stims_names}

        if cells_ids == None:
            cells_ids = self.get_responsive()

        fingerprints = []

        for cell in cells_ids:

            average_resp = self.cells[cell].analyzed_trials

            # concatenate the mean responses to all the trials specified by trial_names,
            # for all the stimuli specified by stim_names.

            concat_stims = []

            for (stim, trials_names) in stim_trials_dict.items():

                if not trials_names:

                    trials_names = list(self.sync.sync_ds[stim])[:-1]

                for trial_name in trials_names:

                    r = average_resp[stim][trial_name]["%s_avg"%rtype]

                    # cut the responses
                    start = average_resp[stim][trial_name]["window"][0]
                    stop = average_resp[stim][trial_name]["window"][1]

                    r = r[start : int(stop + start / 2)]

                    if smooth:
                        # low-pass filter
                        r = filter(r, 0.1)

                    if mode == 'full':
                        concat_stims = np.concatenate((concat_stims, r))

                    elif mode == 'median':
                        r = np.median(r)
                        concat_stims.append(r)              

            concat_stims = np.array(concat_stims)
            
            if mode != 'median':

                if normalize == "linear":

                    concat_stims = lin_norm(concat_stims, -1, 1)

                elif normalize == "zscore":

                    concat_stims = z_norm(concat_stims)

            fingerprints.append(concat_stims)

        # check lenghts consistency
        fingerprints = check_len_consistency(fingerprints)

        # convert to array
        fingerprints = np.array(fingerprints)

        ## NB: index consistency between fingerprints array and 
        # list from get_responsive() is important here!

        return fingerprints

    def get_populations(
        self,
        cells_ids=None,
        algo='pca',
        clusters='kmeans',
        k=None,
        save_name=None,
        tsne_params=None,
        umap_params=None,
        **kwargs
        ):

        '''

        Find functional populations within the set of cells specified. 
        
        - cells_id: list of str
            list of valid cells ids used for identifying which subset of all the cells to analyze.
            By thefault, all the cells present in the recording will be analyzed.
        - algo: str
            algorithm for demensionality reduction. Can be pca or tsne.
        - clusters: str
            clustering algorithm. can be "kmeans" or "gmm" (gaussian mixture model)
        - k: int
            number of expected clusters 
        - **kwargs:
            any valid argument to parametrize compute_fingerprints() method

        '''
        if cells_ids == None:
            cells_ids = self.get_responsive()

        fp = self.compute_fingerprints(
                    cells_ids = cells_ids,
                    **kwargs)
        
        if algo=='pca':
            
            # run PCA
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(fp)
            exp_var = pca.explained_variance_ratio_
            xlabel = "PC1 (% {})".format(round(exp_var[0]*100,2))
            ylabel = "PC2 (% {})".format(round(exp_var[1]*100,2))

        elif algo=='tsne':

            # if needed, go with Tsne
            if tsne_params==None:
                tsne_params =  {
                        'n_components':2, 
                        'verbose':1, 
                        'metric':'cosine', 
                        'early_exaggeration':4, 
                        'perplexity':10, 
                        'n_iter':3000, 
                        'init':'pca', 
                        'angle':0.9}

            transformed = TSNE_embedding(fp,**tsne_params)
            xlabel = "tsne 1"
            ylabel = "tsne 2"

        elif algo=='umap':
            if umap_params==None:
                umap_params =  {
                        'n_components':2, 
                        'n_neighbors':15, 
                        'min_dist': 0.1}
                
            umap = UMAP(**umap_params)
            transformed = umap.fit_transform(fp)
            xlabel = "UMAP 1"
            ylabel = "UMAP 2"

        if k==1:
            labels = np.zeros(len(cells_ids),dtype=int)

        else:
            # clusterize
            if k == None:
                k = find_optimal_kmeans_k(transformed)
                print('Optimal k: ',k)

            if clusters == 'kmeans':
                labels = k_means(transformed, k)
            elif clusters == 'gmm':
                labels = GMM(transformed,n_components=(k),covariance_type='diag')

        fig = plot_clusters(transformed,labels,None,xlabel,ylabel,save=save_name)

        # get popos
        pops = []
        for n in np.unique(labels):

            indices = np.where(labels == n)[0]

            c = []
            for i in indices:

                c.append(cells_ids[i])

            pops.append(c)

        self.populations = pops

        # assign pop label at each cell
        for i,pop in enumerate(pops):
            for id in pop:

                self.cells[id].label = i

        return self.populations, fig
    
    def get_pop_stats(
        self,
        cells_ids, 
        stim_trials_dict=None, 
        rtype='norm'):

        """
        return mean and std of a group of cells
        """
        if stim_trials_dict == None:

            stim_trials_dict = {stim: list(self.sync.sync_ds[stim].keys())[:-1] 
                                for stim in self.sync.stims_names}

        stats = {s:{t:{} for t in stim_trials_dict[s]} for s in stim_trials_dict}

        for s in stim_trials_dict:
            for t in stim_trials_dict[s]:

                avg = []
                for c in cells_ids:

                    r = self.cells[c].analyzed_trials[s][t]['%s_avg'%rtype]
                    avg.append(r)

                avg = np.mean(avg,axis=0)
                sem = np.std(avg,axis=0)/np.sqrt(len(avg))
                stats[s][t] |= {'%s_avg'%rtype:avg, '%s_sem'%rtype:sem} 

        return stats

    def get_neural_states(
        self, 
        cells, 
        stims, 
        trials_types, 
        ntrials, 
        normalize=True,
        median=False,
        avg_trials=False,
        include_peri_act=True,
        get_shuffle=False):

        ntrials_types = len(stims)*len(trials_types)

        # concatenate the PSTHs off all the trials for all the trial types
        psths = []
        for stim in stims:
            for ttype in trials_types:    
                trials = []
                for tix in range(ntrials):
                    trial = []
                    for c in cells:
                        t = c.analyzed_trials[stim][ttype]['trials_norm'][tix][:]
                        if not include_peri_act:
                            s,e = c.analyzed_trials[stim][ttype]['window']
                            t = t[s:e]

                        if normalize:
                            t = z_norm(t)
                        trial.append(t)

                    trial = check_len_consistency(trial)
                    trials.append(trial)

                psths.append(trials)

        psths = np.array(psths)
        if median:
            psths = np.median(psths,axis=-1)[:,:,:,np.newaxis]
        if avg_trials:
            psths = np.mean(psths,axis=1)[:,np.newaxis,:,:]

        print(psths.shape)

        # concatenate trial types
        psths_all = np.concatenate(psths[0],axis=1)
        for i in range(1,ntrials_types):
            psths_ = np.concatenate(psths[i],axis=1)
            psths_all = np.concatenate((psths_all,psths_),axis=1)

        stiml = psths.shape[-1]
        print('trial len: ',stiml)

        if get_shuffle:
            # generate baseline activity from bootstrapping
            baseline = []
            for j in range(ntrials):
                baseline_cells = []
                for i,c in enumerate(cells):
                    baseline_shuff = []
                    for _ in range(50):
                        rand = np.random.choice(c.dff, size=stiml)
                        baseline_shuff.append(rand)

                    baseline_shuff = z_norm(np.mean(baseline_shuff,axis=0))
                    baseline_cells.append(baseline_shuff)

                baseline.append(baseline_cells)

            baseline = np.array(baseline)
            # concatenate trial types for baseline
            baseline_all = np.concatenate(baseline,axis=1)

            # append baseline to psths
            psths_all = np.concatenate((psths_all,baseline_all),axis=1)

        # # concatenate
        # psths = np.concatenate(psths, axis=1)
        print('psths: ',psths_all.shape)
        states = psths_all.T
        print('states: ',states.shape)

        return states, psths, stiml
    
    def _bin_data_(self, binsize, mode='sum'):

        """
        Bin data and sync according to binsize
        """

        if mode == 'sum' : func = np.sum
        if mode == 'mean' : func = np.mean

        print("> Binning sync ...",)

        # sync signal
        sync_s = np.zeros(self.rec_length)
        sync_tps = self.sync.sync_tps

        for i in range(1,len(sync_tps),2):
            sync_s[sync_tps[i-1]:sync_tps[i]] = 1 

        # bin sync
        sync_s_binned = np.array([func(sync_s[int((i-1)*binsize):int((i)*binsize)])
                            for i in tqdm(range(1,int(self.rec_length/binsize)))]).astype(bool)
        
        sync_tps_binned = np.argwhere(np.diff(sync_s_binned)).T[0]
        # save binned sync file .npy
        filename = 'TTL_tp_binned_%dms.npy'%(binsize/self.params['sf']*1000)
        np.save(filename, sync_tps_binned)
        # generate new sync object
        new_sync =  Sync().generate_data_structure(filename,
                                                    self.loader.stim_dict_file,
                                                    self.loader.trials_names)
        self.sync = new_sync
        print('OK')

        print("> Binning data ...")
        data_binned = []
        for j in tqdm(range(1,int(self.rec_length/binsize))):

            start = int((j-1)*binsize)
            stop = int(j*binsize)
            data_binned.append(func(self.loader.dataRaw[:,start:stop], axis=1))
            
        data_binned = np.array(data_binned).T
        print('OK')

        self.dataRaw = data_binned

    def _extract_dataBehav_(self, dataBehav_names=[]):

        """ 
        Extract the supllementary data (e.g pupil, treadmill ecc) if present
        """
        if not dataBehav_names:
            dataBehav_names = list(self.dataBehav.keys())
                
        for data_name in dataBehav_names:

            data = self.dataBehav[data_name]
            self.dataBehav_analyzed |= {data_name:{}}
            print("\n> Analyzing %s data ... "%data_name)

            # check if the end of the last stim_window specifyed by the sync structure exceeds 
            # the length of the recording. If not, pad the recording. This can be due to premature 
            # end of the recording, where the pause after the last trial is too short.
            if self.sync.sync_ds[self.sync.stims_names[-1]]['stim_window'][1]>self.rec_length:
                    
                    pad_len = self.sync.sync_ds[self.sync.stims_names[-1]]['stim_window'][1]-self.rec_length

                    data = np.pad(data,((0,0),(0,pad_len)),mode='constant',
                                    constant_values=((0,0),(0,np.mean(data[-10:]))))
                    
                    warnings.warn('Last trial has been padded (pad length:%d, data value:%d)'%(pad_len,
                                np.mean(data[-1])), RuntimeWarning)

            # EXTRACT TRIALS AND COMPUTE STATISTICS #

            for stim in self.sync.stims_names:

                self.dataBehav_analyzed[data_name] |= {stim:{}}
                print('\n  Extracting stim: %s'%stim)


                if self.params["baseline_extraction_behavData"] == 1:

                    # extract only one baseline and std for each stimulus
                    baseline = data[self.sync.sync_ds[stim]["stim_window"][0] - 
                                        int(self.params["baseline_length"]*self.sf) : 
                                        self.sync.sync_ds[stim]["stim_window"][0]]

                    mean_baseline = np.mean(baseline)
                    std = np.std(baseline)
                
                # get max value whithin the stimulus block
                stim_max = data[self.sync.sync_ds[stim]['stim_window'][0]:
                                self.sync.sync_ds[stim]['stim_window'][1]].max()

                pbar = tqdm(list(self.sync.sync_ds[stim].keys())[:-1]) # last item is stim_window
                for trial_type in pbar:  

                    pbar.set_description('    extracting trial: %s'%trial_type)
                    trials_raw = []
                    trials_norm = []

                    trial_len = self.sync.sync_ds[stim][trial_type]["trial_len"]
                    pause_len = self.sync.sync_ds[stim][trial_type]["pause_len"]

                    # if pause_len > self.params["max_aftertrial"]*self.sf:

                    #     pause_len = int(self.params["max_aftertrial"]*self.sf)

                    if self.params["max_aftertrial"]*self.sf>=0:

                        pause_len = int(self.params["max_aftertrial"]*self.sf)

                    # EXTRACT AND NORMALIZE each trial independently#
                    for trial in self.sync.sync_ds[stim][trial_type]["trials"]:

                        d_trial = data[trial[0]- int(self.params["pre_trial"]*self.sf) :
                                            trial[0] + trial_len + pause_len].copy()

                        if self.params["baseline_extraction_behavData"] == 0:

                            # extract local baselines for each trial 
                            baseline = data[trial[0] - int(self.params["baseline_length"]*self.sf) : trial[0]]
                            mean_baseline = np.mean(baseline)    
                            std = np.std(baseline)
                        
                        # Normalize: subract baseline
                        d_trial_norm = (d_trial - mean_baseline)
                        # Normalize: rescale between 0 and 1 within the stimulus block
                        # d_trial_norm = d_trial/stim_max
                    
                        trials_raw.append(d_trial)
                        trials_norm.append(d_trial_norm)

                    # COMPUTE STATISTICS #
                    trials_raw = np.array(trials_raw)
                    trials_norm = np.array(trials_norm)

                    if trials_raw.shape[0] > 1:

                        trials_raw_avg = np.mean(trials_raw, axis=0)
                        trials_raw_std = np.std(trials_raw, axis=0)
                        trials_norm_avg = np.mean(trials_norm, axis=0)
                        trials_norm_med = np.median(trials_norm, axis=0)
                        trials_norm_std = np.std(trials_norm, axis=0)

                    else:

                        trials_raw_avg = trials_raw[0]
                        trials_raw_std = 0
                        trials_norm_avg = trials_norm[0]
                        trials_norm_med = trials_norm[0]
                        trials_norm_std = 0

                    on = int(self.params["pre_trial"]*self.sf)
                    off = int(self.params["pre_trial"]*self.sf + trial_len)

                    # store data

                    self.dataBehav_analyzed[data_name][stim] |= {
                            trial_type: {
                                "raw_avg":trials_raw_avg,
                                "raw_err": trials_raw_std,
                                "trials_raw": trials_raw,
                                "trials_norm": trials_norm,
                                "norm_avg": trials_norm_avg,
                                "norm_med": trials_norm_med,
                                "norm_err": trials_norm_std,
                                "window": (on, off),
                            }
                        }


        return self.dataBehav_analyzed
        
    def _extract_data_(self, keep_unresponsive: bool=False):

        """
        Extract the Data fro all the cells present in the recording files.

        - keep_unresponsive: bool
            decide wether to keep also the cells classified as unresponsive using
            the criteria and the threshold specified in thge config file.
        """

        if self.dataRaw.size > 0:

            print("\n> Analyzing neural data ... ")

            # check if the end of the last stim_window specifyed by the sync structure exceeds 
            # the length of the recording. If not, pad the recording. This can be due to premature 
            # end of the recording, where the pause after the last trial is too short.
            if self.sync.sync_ds[self.sync.stims_names[-1]]['stim_window'][1]>self.rec_length:

                    pad_len = self.sync.sync_ds[self.sync.stims_names[-1]]['stim_window'][1]-self.rec_length

                    self.dataRaw = np.pad(self.dataRaw,((0,0),(0,pad_len)),mode='constant',
                                    constant_values=((0,0),(0,np.mean(self.dataRaw[-10:]))))
                    
                    warnings.warn('Last trial has been padded (pad length:%d, data value:%d)'%(pad_len,
                                np.mean(self.dataRaw[-1])), RuntimeWarning)

            # EXTRACT TRIALS AND COMPUTE STATISTICS #

            # instantiate cell objects
            cells = [Cell(self, i) for i in range(self.ncells)]

            # initilaize best quality indices
            best_qis = np.zeros(self.ncells)

            for stim in self.sync.stims_names:

                print('\n  Extracting stim: %s'%stim)

                for c in cells: c.analyzed_trials.update({stim: {}})

                if self.params["baseline_extraction_neuData"] == 1:

                    # extract only one baseline for each stimulus
                    if self.params["baseline_indices"] is not None:

                        mean_baselines = np.mean(
                                    self.dataRaw[:, self.params["baseline_indices"][0]: 
                                                self.params["baseline_indices"][1]], axis=1)
                    else:
                        
                        mean_baselines = np.mean(
                                    self.dataRaw[:,self.sync.sync_ds[stim]["stim_window"][0] - 
                                                int(self.params["baseline_length"]*self.sf) : 
                                                self.sync.sync_ds[stim]["stim_window"][0]]
                                        , axis=1)

                pbar = tqdm(list(self.sync.sync_ds[stim].keys())[:-1]) # last item is stim_window
                for trial_type in pbar :  
                    
                    pbar.set_description('    extracting trial: %s'%trial_type)

                    trials_raw = []
                    trials_norm = []

                    trial_len = self.sync.sync_ds[stim][trial_type]["trial_len"]
                    pause_len = self.sync.sync_ds[stim][trial_type]["pause_len"]


                    # if pause_len > self.params["max_aftertrial"]*self.sf:

                    #     pause_len = int(self.params["max_aftertrial"]*self.sf)

                    if self.params["max_aftertrial"]*self.sf>=0:

                        pause_len = int(self.params["max_aftertrial"]*self.sf)

                    # EXTRACT AND NORMALIZE DATA for each trial#
                    for trial in self.sync.sync_ds[stim][trial_type]["trials"]:

                        if self.params["baseline_extraction_neuData"] == 0:

                            # extract local baselines for each trial 
                            mean_baselines = np.mean(self.dataRaw[:
                                                    ,trial[0] - int(self.params["baseline_length"]*self.sf) : trial[0]]
                                                    ,axis=1)

                        resps = self.dataRaw[:,trial[0]- int(self.params["pre_trial"]*self.sf) :
                                            trial[0] + trial_len + pause_len].copy()
                        
                        trials_raw.append(resps)
                        
                        # if data contains fluo traces, compute df/f normalization
                        if self.dtype == 'fluo':
                            resps_dff = ((resps.T - mean_baselines) / mean_baselines).T
                            trials_norm.append(resps_dff)

                        # if data contains spiking activity, compute z-score 
                        elif self.dtype == 'spikes':
                            # resps_std = np.std(resps, axis=1)
                            # resps_zscore = ((resps.T - mean_baselines)/resps_std).T
                            resps_zscore = z_norm(resps)
                            trials_norm.append(resps_zscore)

                    # COMPUTE STATISTICS #
                    trials_raw = np.array(trials_raw)
                    trials_norm = np.array(trials_norm)
                    

                    if trials_raw.shape[0] > 1:
                        trials_raw_avg = np.mean(trials_raw, axis=0)
                        trials_raw_std = np.std(trials_raw, axis=0)
                        trials_norm_avg = np.mean(trials_norm, axis=0)
                        trials_norm_std = np.std(trials_norm, axis=0)

                    else:
                        trials_raw_avg = trials_raw[0]
                        trials_raw_std = np.zeros(self.ncells)
                        trials_norm_avg = trials_norm[0]
                        trials_norm_std = np.zeros(self.ncells)
                    
                    # calculate quality indices over the trials and udate the best qis
                    # PENDING: implementation of ttest-based qi

                    qis = []
                    for cix in range(self.ncells):
                        filtered = filter(trials_norm[:,cix,:], 0.3)
                        zscored = z_norm(filtered)
                        if self.params['qi_metrics'] == 0 and trials_norm.shape[0] > 1 and np.any(trials_norm):
                            qis.append(compute_QI(zscored, (int(self.params['pre_trial']*self.sf),-1)))
                        else:
                            # calculate responsiveness based on zscore
                            qis.append(np.abs(z_norm(trials_norm_avg[cix,:])).max())

                    best_qis = np.where(np.greater(qis,best_qis), qis, best_qis)

                    # else: qis = None

                    on = int(self.params["pre_trial"]*self.sf)
                    off = int(self.params["pre_trial"]*self.sf + trial_len)

                    # store cell data in each cell object
                    for i,c in enumerate(cells): 

                        c.analyzed_trials[stim] |= {
                                trial_type: {
                                    "raw_avg":trials_raw_avg[i],
                                    "raw_std": trials_raw_std[i],
                                    "trials_norm": trials_norm[:,i,:],
                                    "trials_raw": trials_raw[:,i,:],
                                    "norm_avg": trials_norm_avg[i],
                                    "norm_std": trials_norm_std[i],
                                    "qi": qis[i],
                                    "window": (on, off),
                                }
                            }
                        c.norm = self.dataNorm[i]
                        c.raw = self.dataRaw[i]
    
            # store all the cells in a dictionary with id:cell items
            self.cells = {idx:cell for idx,cell in enumerate(cells)}

            # asses responsiveness for each cell
            for i,c in list(self.cells.items()): 

                if best_qis[i] != 0:
                    c.qi = best_qis[i]
                    c.is_responsive()

                # else:
                #     # if was impossible to compute QI because the data contain only
                #     # a single trial for every stimulus, assume is responsive
                #     c.qi = None
                #     c.responsive = True 

                if not (keep_unresponsive or c.responsive):

                    self.cells.pop(i)

            # order cells according to QI
            qis = [c.qi for c in self.cells.values()]
            sorted_ix = np.argsort(qis)[::-1]
            self.cells = {list(self.cells.keys())[i]: self.cells[list(self.cells.keys())[i]] for i in sorted_ix}

            # retrive the indices of responsive cells
            self.responsive = self.get_responsive()

            if not self.responsive:

                warnings.warn("No responsive cells found!", RuntimeWarning)

            else:
        
                print(
                    "\n> %d responsive cells found (tot: %d, keep_unresponsive: %r, use_iscell: %r)"
                    % (
                        len(self.responsive),
                        self.ncells,
                        keep_unresponsive,
                        self.params["use_iscell"],
                    )
                )

            return self.cells
