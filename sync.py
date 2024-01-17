import numpy as np
from scipy import io
import os
import json


class Sync:

    """
    A Sync object that allows to allign a Record object to a Recording object.
    """

    def __init__(self):

        """
        Initialize synchroniuzer object
        """

        self.sync_tps = []
        self.stims_names = []
        self.trials_names = None
        self.sync_ds = None

    def generate_data_structure(
        self,
        sync_file: str,
        stims_sequence_file: str,
        trials_names: dict = None):

        """
        Create a sync data structure from scanbox metadata file.
        It requires also a .npy file where the full sequence of stimuli is stored.
        This class is ment to work with recordings where different types of stimuli
        were presented to either ipsilateral, controlateral or both eyes, and the tps
        corresponding to the onset and offset of each trial stored in a .mat file.

        - sync_file:
            .mat or .npy file containing a sequence of tps. Starting from the first element,
            every pair of tps will be considered as the onset and the offset of the trials (or events) specified
            in the stims_sequence_file.
        - stims_sequence_file:
            absolute path to .json file containing the full sequence of stimuli presented
            during the recording.
            The dict should contain pairs (stim_name:[trial_a1,...,trial_z1,...,trial_an,...,trial_zn])
            The different trial types (or events) should be encoded as a sequence of int values, fro 0 to (#trials types -1)
        - stim_names:
            list of strings containing the names of each stimuli type. It is assumed to to have a
            dimension equal to the first dimension of the data structure decribed by the stims_sequence_file,
            and to have the correct order.
        -trials_names:
            dict containing the names of the trials associated to the values contained in the stim_sequence_file,
            in the form: {value_0 : "trial_name_0", ... , value_n : "trial_name_n"}. All the in values contained
            in the stim_sequence_file must be associated to a trial name.

        """
        self.__init__()

        # read sync file
        ext = os.path.splitext(sync_file)[1]
        if ext == '.mat':
            self.sync_tps = np.squeeze(io.loadmat(sync_file)["info"][0][0][0])
        elif ext == '.npy':
            self.sync_tps = np.load(sync_file).astype(int)

        # read stim dict
        with open(stims_sequence_file, "r") as f:

            self.stims_sequence = json.load(f) 

        self.stims_names = list(self.stims_sequence.keys())
        
        if trials_names == None:
            trials_names = {}

        """
        Generate a data structure where to store all the onset and offset tps
        for each trial type, for each stimulus type
        """

        sync_ds = {}
        stim_dict = {}

        # retrive all the (on_frame,off_frame) for all the trials
        i = 0
        for stim in self.stims_sequence:

            sync_ds |= {stim: {}}
            stim_dict |= {stim: []}

            sequence = self.stims_sequence[stim]
            stim_start = self.sync_tps[i]

            for trial_type in np.unique(sequence):

                trial_len = self.sync_tps[i + 1] - self.sync_tps[i]

                if (i+2) < len(self.sync_tps):
                    pause_len = self.sync_tps[i + 2] - self.sync_tps[i + 1]
                else: 
                    pause_len = 0

                if trial_type in trials_names:
                    trial_name = trials_names[trial_type]

                else: 
                    trial_name = trial_type
                    trials_names |= {trial_type:trial_type}
                    
                sync_ds[stim] |= {
                    trial_name: {
                        "trials": [],
                        "trial_len": trial_len,
                        "pause_len": pause_len,
                    }
                }

                stim_dict[stim].append(trial_name)

            for trial_type in sequence:

                if trials_names != None:
                    trial_name = trials_names[trial_type]
                else: 
                    trial_name = trial_type

                sync_ds[stim][trial_name]["trials"].append(
                    (self.sync_tps[i], self.sync_tps[i + 1])
                )

                i += 2

            stim_end = (self.sync_tps[i - 1]+
                        sync_ds[stim][trial_name]["pause_len"])

            sync_ds[stim] |= {"stim_window": (stim_start, stim_end)}
        
        self.sync_ds = sync_ds
        self.stim_dict = stim_dict
        self.trials_names = trials_names

        return self

    def load_data_structure(
            self, 
            ds_json):

        """
        Load data structure from a json file.
        See the example sync_dict_example.json for the structure that the input ds_json file must have.
        """
        self.__init__()

        with open(ds_json, "r") as f:

            self.sync_ds = json.load(f)


        for stim in self.sync_ds:

            self.stims_names.append(stim)

            for i,trial_type in enumerate(self.sync_ds[stim]):
            
                if trial_type not in self.trials_names.values():

                    self.trials_names |= {i:trial_type}

                for trial in self.sync_ds[stim][trial_type]["trials"]:

                    self.sync_tps.extend(trial)

        return self


    def generate_data_structure_sn(
        self,
        sync_file: str,
        texture_file: str,
        trial_len=20,
        ):

        """
        Crete a sync data structure specifically for Sparse Noise recordings.
        It requires the scanbox metadata .mat file and the sparse noise texture .mat file
        
        - sync_file:
            .mat file containing a sequence of tps. Starting from the first element,
            every pair of tps will be considered as the onset and the offset of the trials (or events) specified
            in the stims_sequence_file.
        - texture_file:
            .mat file containing the sequence of matrices which represent the textures presented during the sparse noise
            stimulation.
        - trial_len: int
            number of tps that will be considered as part of a trial, starting from the onset frame present in sync_file
        """     

        self.sync_tps = np.squeeze(io.loadmat(sync_file)["info"][0][0][0])
        self.textures = io.loadmat(texture_file)['stimulus_texture']
        self.text_dim = self.textures.shape
        self.trial_len = trial_len
        self.trials_names = []
        self.stims_names.append("sparse_noise")
        self.sync_ds = {'sparse_noise':{}}

        for i in range(self.text_dim[1]):
            for j in range(self.text_dim[2]):

                on_indexes = np.where(self.textures[:,i,j]==1)[0]
                off_indexes = np.where(self.textures[:,i,j]==0)[0]

                # extract sync tps where square turned white
                on_tps = [(frame,frame+trial_len) for frame in self.sync_tps[on_indexes]]
                # extract sync tps where square turned black
                off_tps = [(frame,frame+trial_len) for frame in self.sync_tps[off_indexes]]

                # on trial
                self.trials_names.append('%d_%d_on'%(i,j))
                self.sync_ds['sparse_noise']|=({'%d_%d_on'%(i,j): 
                                                {'trials': on_tps,
                                                'trial_len': trial_len,
                                                'pause_len': 0}})
                # off trial 
                self.trials_names.append('%d_%d_off'%(i,j))
                self.sync_ds['sparse_noise']|=({'%d_%d_off'%(i,j): 
                                                {'trials': off_tps,
                                                'trial_len': trial_len,
                                                'pause_len': 0}})




        self.sync_ds['sparse_noise']|={'stim_window':(self.sync_tps[0],self.sync_tps[-1])}

        return self  
    
    def minimal_data_structure(self, ntps):

        """
        Generate a default minimalistic sync structure with one stimulus and 
        one trial
        """

        pass

