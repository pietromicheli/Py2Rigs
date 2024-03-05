import numpy as np
from .rec import Rec 
from .utils import *
from .plot import plot_clusters
from sklearn.decomposition import PCA
from umap import UMAP

class Batch:

    """
    Class for performing batch analysis of multiple recordings.
    All the recordings contained in a Batch2P object must share at
    least one stimulation condition with at least one common trial
    type (e.g. CONTRA, BOTH or IPSI)

    """

    def __init__(self, loaders:list):

        """
        Create a Batch2P object from a data dictionary.

        - loaders
            A list of Dataloaders containing the data and the sync of each recording.
        """

        # INSTANTIATE REC OBJECTS #

        self.recs = {}
        recs_list = []
        groups_list = []
        for rec_id,ld in enumerate(loaders):

            print("\n> creating Rec obj with index: %d:"%rec_id)
            rec = Rec(ld,id=rec_id)
            self.recs |= {rec_id:rec}
            recs_list.append(rec)
            groups_list.append(rec.group)

        groups_list = list(set(groups_list))
        ## group recs
        self.recs_groups = {g:[] for g in groups_list}
        for r in recs_list: self.recs_groups[r.group].append(r.id)
        
        # RETRIVE SUPPLEMENTARY DATA #
            
        ### for the data present in the  dataSupp of all the recording, keep only the shared ones.
        # dataSupp = [list(rec.dataSupp.keys()) for rec in recs_list]

        ## start from minimal stim set
        # dataSupp_intersection = set(dataSupp[np.argmin([len(s) for s in dataSupp])])

        # for ds in dataSupp[1:]:

        #     dataSupp_intersection.intersection(set(ds))

        # self.dataSupp_intersection = list(dataSupp_intersection)

        ### or keep all the dataSupp available for from each recordings

        self.dataSupp = []
        for rec in recs_list: 
            self.dataSupp.extend(list(rec.dataSupp.keys()))
        self.dataSupp = np.unique(self.dataSupp)

        # RETRIVE STIMULATION PROTOCOLS #
            
        ## be sure that the sync object (if present) of all the recordings share at least 1 stimulus.
        ## only shared stimuli will be used.

        self.stims_trials_intersection = {}
        stims_allrec = [rec.sync.stims_names for rec in recs_list]

        ## start from minimal stim set
        stims_intersection = set(stims_allrec[np.argmin([len(s) for s in stims_allrec])])

        for stims in stims_allrec[1:]:

            stims_intersection.intersection(set(stims))

        ## also, for all the shared stimuli,
        ## select only trials type are shared for that specific stimulus by all recs.

        for stim in list(stims_intersection):

            ## start with trials for stimulus "stim" in the sync_ds of first loader
            trials = list(recs_list[0].sync.sync_ds[stim].keys())[:-1]

            trials_intersection = set(trials)

            for rec in recs_list[1:]:

                trials = list(rec.sync.sync_ds[stim].keys())[:-1]
                ## last item is "window_len"
                trials_intersection.intersection(set(trials))

            self.stims_trials_intersection |= {stim: list(trials_intersection)}

        self.cells = None
        self.populations = []

    def load_params(self):

        """
        Read parameters from .yaml params file

        """

        for rec in self.recs:
            
            print("\n> Rec %d:"%rec)
            self.recs[rec].load_params()

    def extract_all(self, keep_unresponsive=False):

        """
        Extract neural and supplementary data from all the recordings
        """
        
        suppData = self._extract_dataSupp_()
        cells = self._extract_data_(keep_unresponsive=keep_unresponsive)
        
        return cells
    
    def get_responsive(self):

        """
        Get a list containing the ids of all the responsive cells

        """

        ids = []

        for cell in self.cells:

            if self.cells[cell].responsive:

                ids.append(cell)

        return ids

    def compute_fingerprints(
        self, 
        cells_ids=None,
        stim_trials_dict=None, 
        rtype="norm", 
        normalize="zscore", 
        smooth=True):

        """
        Compute a fingerprint for each cell by concatenating the average responses
        to the specified stimuli and trials.

        - cells_ids: list of valid ids
            by default, compute fingerprints of all the responsive cells
        - stim_trials_dict: dict
            A dict which specifies which stim and which trials to concatenate for computing
            the fingerptint.
            Should contain key-values pairs such as {stim:[t1,...,tn]}, where stim is a valid
            stim name and [t1,...,tn] is a list of valid trials for that stim.
        """

        # check if the cells have already been retrived

        if self.cells == None:

            self.get_cells()

        if stim_trials_dict == None:

            stim_trials_dict = {stim: [] for stim in self.stims_trials_intersection}

        responsive = self.get_responsive()

        fingerprints = []

        if cells_ids == None:
            
            cells_ids = responsive

        for cell in cells_ids:

            average_resp = self.cells[cell].analyzed_trials

            # concatenate the mean responses to all the trials specified by trial_names,
            # for all the stimuli specified by stim_names.

            concat_stims = []

            for (stim, trials_names) in stim_trials_dict.items():

                if not trials_names:

                    trials_names = list(self.stims_trials_intersection[stim])

                for trial_name in trials_names:

                    r = average_resp[stim][trial_name]["%s_avg" %rtype]

                    # cut the responses
                    start = average_resp[stim][trial_name]["window"][0]
                    stop = average_resp[stim][trial_name]["window"][1]

                    r = r[start : int(stop + start / 2)]

                    if smooth:
                        # low-pass filter
                        r = filter(r, 0.3)

                    concat_stims = np.concatenate((concat_stims, r))

            if normalize == "linear":

                concat_stims = lin_norm(concat_stims, -1, 1)

            elif normalize == "zscore":

                concat_stims = z_norm(concat_stims)

            fingerprints.append(concat_stims)

        # check lenghts consistency
        fingerprints = check_len_consistency(fingerprints)

        # convert to array
        fingerprints = np.array(fingerprints)

        ## NB: index consistency between fingerprints array and list from get_responsive() is important here!

        return fingerprints

    def get_populations(
        self,
        cells_ids=None,
        algo='pca',
        clusters='kmeans',
        k=None,
        markers=True,
        save_name=None,
        groups_names=None,
        tsne_params=None,
        umap_params=None,
        marker_mode=1,
        **kwargs
        ):

        '''

        Find functional populations within the set of cells specified. 
        
        - cells_id: list of str
            list of valid cells ids used for identify which subset of all the cells to analyze.
            By thefault, all the cells present in the batch will be analyzed.
        - algo: str
            algorithm for demensionality reduction. Can be pca or tsne.
        - clusters: str
            clustering algorithm. can be "kmeans" or "gmm" (gaussian mixture model)
        - k: int
            number of expected clusters.
        - marker_mode: int
            0: markers represent the groups
            1: markers represent the recordings
        - **kwargs:
            any valid argument to parametrize compute_fingerprints() method

        '''

        if cells_ids == None: cells_ids = list(self.cells.keys())

        if groups_names == None: 
            if marker_mode == 0: groups_names = list(self.recs_groups.keys())
            else: groups_names = list(self.recs.keys())

        fp = self.compute_fingerprints(
                    cells_ids = cells_ids,
                    **kwargs)
        
        if algo=='pca':
            
            # run PCA
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(fp)
            exp_var = pca.explained_variance_ratio_
            xlabel = "PC1 ({} %)".format(round(exp_var[0]*100,2))
            ylabel = "PC2 ({} %)".format(round(exp_var[1]*100,2))

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

        # clusterize
        if k == None:
            k = find_optimal_kmeans_k(transformed)

        if clusters == 'kmeans':
            labels = k_means(transformed, k)
        elif clusters == 'gmm':
            labels = GMM(transformed,n_components=(k),covariance_type='diag')

        if markers:
            markers = [int(id.split(sep='_')[marker_mode]) for id in cells_ids]

        else:
            markers=None

        plot_clusters(transformed,labels,markers,xlabel,ylabel,groups_names=groups_names,save=save_name)

        # get popos BATCH
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

        # get popos for each REC
        for rec in self.recs.values():
            pops_rec = []

            for i in np.unique(labels):
                pop = []
                for id,c in rec.cells.items():
                    if c.label == i:
                        pop.append(id)

                pops_rec.append(pop)
            
            rec.populations = pops_rec


        return self.populations
   
    def get_pop_stats(
        self,
        cells_ids, 
        stim_trials_dict=None, 
        rtype='norm'
        ):

        """
        return mean and std of a group of cells
        """
        if stim_trials_dict == None:

            stim_trials_dict = self.stims_trials_intersection

        stats = {s:{t:{} for t in stim_trials_dict[s]} for s in stim_trials_dict}

        for s in stim_trials_dict:
            for t in stim_trials_dict[s]:

                avg = []
                for c in cells_ids:

                    r = self.cells[c].analyzed_trials[s][t]['%s_avg'%rtype]
                    avg.append(r)

                avg = check_len_consistency(avg)
                avg = np.mean(avg,axis=0)
                sem = np.std(avg,axis=0)/np.sqrt(len(avg))
                stats[s][t] |= {'%s_avg'%rtype:avg, '%s_sem'%rtype:sem} 

        return stats

    def _extract_data_(self, keep_unresponsive=False):

        """
        Extract the neuaral data from all cells in each recording and assign new ids.
        Id is in the form G_R_C, where G,R and C are int which specify the group,
        the recording and the cell ids.

        """

        self.cells = {}
        groups = []
        for (rec_id, rec) in self.recs.items():
            
            print('\nRec %d :'%rec_id)

            group_id = rec.group
            groups.append(group_id)

            # retrive cells from each recording
            rec._extract_data_(keep_unresponsive)

            for id,cell in rec.cells.items():

                # new id
                new_id = "%s_%s_%s" % (str(group_id), str(rec_id), str(id))
                self.cells |= {new_id: cell}
                # # update cell id
                # cell.id = new_id

        # RETRIVE THE GROUPPED CELLS
        self.cells_groups = {g:{} for g in set(groups)}

        for id,cell in self.cells.items():
            for g in self.cells_groups:

                if int(id.split('_')[0])==g:

                    self.cells_groups[g] |= {id:cell}

        return self.cells

    def _extract_dataSupp_(self):

        """
        Extract the supplementary data (eye tracking, tredmill ecc.) from each recording.
        """

        self.dataSupp_analyzed = {ds:{} for ds in self.dataSupp}
            
        for (rec_id, rec) in self.recs.items():

            print('\nRec %d :'%rec_id)

            if not rec.dataSupp: continue
            rec._extract_dataSupp_(dataSupp_names=list(rec.dataSupp.keys()))

            for dataSupp_name in self.dataSupp:

                self.dataSupp_analyzed[dataSupp_name] |= {rec.id:rec.dataSupp_analyzed[dataSupp_name]}

        # extract averages across recs
        if len(self.recs)>1:
                
            for dataSupp_name in self.dataSupp:

                # prepare dict
                batch_stat = {'batch_stat':{s:{t:{} for t in self.stims_trials_intersection[s]} for s in self.stims_trials_intersection}}

                self.dataSupp_analyzed[dataSupp_name] |= batch_stat

                for stim in self.stims_trials_intersection:
                    for trial in self.stims_trials_intersection[stim]:
                        
                        avg_norm = []
                        std_norm = []
                        avg_raw = []
                        std_raw = []
                        for (rec_id, rec) in self.recs.items():

                            if dataSupp_name not in rec.dataSupp:
                                continue
                            
                            dataSupp = rec.dataSupp_analyzed[dataSupp_name]
                            avg_norm.append(dataSupp[stim][trial]['norm_avg'])
                            std_norm.append(dataSupp[stim][trial]['norm_std'])
                            avg_raw.extend(dataSupp[stim][trial]['trials_raw'])
                            std_raw.append(dataSupp[stim][trial]['raw_std'])

                        avg_norm = check_len_consistency(avg_norm)
                        std_norm = check_len_consistency(std_norm)
                        avg_raw = check_len_consistency(avg_raw)
                        std_raw = check_len_consistency(std_raw)
                        # sem = np.std(avg, axis=0)/np.sqrt(len(avg))
                        avg_norm = np.mean(avg_norm,axis=0)
                        std_norm = np.mean(std_norm,axis=0)
                        avg_raw = np.mean(avg_raw,axis=0)
                        std_raw = np.mean(std_raw,axis=0)

                        batch_stat['batch_stat'][stim][trial] |= {'norm_avg':avg_norm, 'norm_std':std_norm,  
                                                                'raw_avg':avg_raw, 'raw_std':std_raw, 
                                                                'window':dataSupp[stim][trial]['window']}

                self.dataSupp_analyzed[dataSupp_name] |= batch_stat

        return self.dataSupp_analyzed
