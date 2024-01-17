from Py2Rigs import sync
import os
import numpy as np
import warnings
import json

class Dataloader():

    def __init__(self, group:int=0):
        
        self.dype = None
        self.dataRaw = np.empty(0)
        self.sync = None
        self.dataSupp = {}
        self.group = group
        self.dtype = None
        self.rec_length = None
        self.ncells = None

    def load_sync(self, sync_file, stim_dict_file, trials_names=None):
        
        self.sync_file = sync_file
        self.stim_dict_file = stim_dict_file
        self.trials_names = trials_names

        self.sync = sync.Sync().generate_data_structure(sync_file, stim_dict_file, trials_names)
        print("> Sync object generated")

    def load_fluo(self, F_path, Fneu_path = None):

        '''
            Load data from the output files generated by Suite2p program.
            
            - s2p_dir_path: str
                absolute path to the directory containing the files generated by Suite2p.
                
        '''
    
        self.dtype = 'fluo'
        
        supported_fmt = ['.csv', '.npy']

        paths = [F_path]

        if Fneu_path != None:
            paths.append(Fneu_path)

        for d_path in paths:

            ext = os.path.splitext(d_path)[1]

            if ext not in supported_fmt:

                raise Exception("ERROR: please provide a data path to a supported file.",
                                "\nSupported formats are: {} {}".format(*supported_fmt))
            
            # load data
            print("> loading data from %s ..." % d_path, end=" ")

            if ext == '.npy':
                data = np.load(d_path)
            elif ext == '.csv':
                data = np.loadtxt(d_path, delimiter=',', dtype=np.float64, skiprows=1)

            self.dataRaw = data

            if not isinstance(self.dataRaw[0],np.ndarray):
                self.dataRaw = np.array([self.dataRaw])

            self.rec_length = self.dataRaw.shape[1]            
            self.ncells = self.dataRaw.shape[0]

            print("Loaded arrays of size %dx%d\n"%(self.ncells, self.rec_length))

        return self

    def load_s2p_dir(self, s2p_dir_path):

        self.dtype = 'fluo'

        if not os.path.isdir(s2p_dir_path):

            raise Exception("ERROR: Please provide a valid data path to a suite2p output direcory")
        
        files = os.listdir(s2p_dir_path)

        # F.npy is mandatory
        if "F.npy" not in files:

            raise Exception("ERROR: F.npy not found in %s" % (s2p_dir_path))

        # load data
        print("\n> loading data from %s ..." % s2p_dir_path, end=" ")

        self.dataRaw = np.load(s2p_dir_path + r"/F.npy")

        try:
            self.Fneu = np.load(s2p_dir_path + r"/Fneu.npy")
        except:
             warnings.warn("WARNING:",
             "Fneu.npy not found in %s"%s2p_dir_path, RuntimeWarning)
        try:
            self.spks = np.load(s2p_dir_path + r"/spks.npy")
        except:
             warnings.warn("WARNING:",
              "spks.npy not found in %s"%s2p_dir_path, RuntimeWarning)
        try:
            self.iscell = np.load(s2p_dir_path + r"/iscell.npy")
        except:
             warnings.warn("WARNING:",
              "iscell.npy not found in %s"%s2p_dir_path, RuntimeWarning)
        try:
            self.stat = np.load(s2p_dir_path + r"/stat.npy", allow_pickle=True)
        except:
             warnings.warn("WARNING:",
              "stat.npy not found in %s"%s2p_dir_path, RuntimeWarning)
        try:
            self.ops = np.load(s2p_dir_path + r"/ops.npy", allow_pickle=True)
        except:
             warnings.warn("WARNING:",
              "ops.npy not found in %s"%s2p_dir_path, RuntimeWarning)

        self.rec_length = self.dataRaw.shape[1]
        self.ncells = self.dataRaw.shape[0]

        print("Loaded data of size %dx%d\n"%(self.ncells, self.rec_length))

        return self
    
    def load_spikes(self, spks_path):

        """
        Load pre-proccesed ephys data from .npy file.
        Every row coontains the spiking activity (spike counts or spiking rates) of a single unit.
        """

        self.dtype = 'spikes'

        ext = os.path.splitext(spks_path)[1]

        self.s2p_dir_path = spks_path

        if ext != '.npy':

            raise Exception("ERROR: Please provide a data path to a .npy file.")

        print('> loading data ...', end='')
        self.dataRaw = np.load(spks_path)
        self.rec_length = self.dataRaw.shape[1]
        self.ncells = self.dataRaw.shape[0]

        print('OK')

        return self

    def load_supp(self, dataSupp_path, data_name):

        """
        Load supplementary 1D data, such as tracked pupil area, tradmill, ecc.
        """

        ext = os.path.splitext(dataSupp_path)[1]
        if ext != '.npy':

            raise Exception("ERROR: Please provide a data path to a .npy file.")

        print('> loading %s ...'%data_name, end='')
        data = np.load(dataSupp_path)
        self.dataSupp |= {data_name: data}
        print('OK')

        if self.rec_length == None: self.rec_length = data.size

