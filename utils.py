import os
import shutil
import pathlib
import h5py
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import butter, filtfilt, sosfiltfilt, lfilter
from scipy.optimize import curve_fit, minimize
from scipy.fftpack import rfft, irfft, rfftfreq
from sklearn.manifold import TSNE, MDS
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

#############################
###---UTILITY FUNCTIONS---###
#############################

CONFIG_FILE_TEMPLATE = r"%s/params.yaml" % pathlib.Path(__file__).parent.resolve()
DEFAULT_PARAMS = {}


def generate_params_file():

    """
    Generate a parameters file in the current working dir.
    This file contains a list of all the parameters that will used for the downsteream analysis,
    set to a default value.
    """

    files = os.listdir(os.getcwd())

    if "params.yaml" not in files:

        print("> Config file generated. All parameters set to default.")

        return shutil.copy(CONFIG_FILE_TEMPLATE, "params.yaml")

    else:

        print("> Using the parameters file found in current dir.")

        return "params.yaml"

def save_dict_to_hdf5(dic, filename):

    def _save_dict_to_hdf5(h5file, path, dic):

        for key, item in dic.items():

            if isinstance(item, dict):
                _save_dict_to_hdf5(h5file, f"{path}/{str(key)}", item)

            else:
                try:
                    h5file[path + '/' + str(key)] = item
                except:
                    # save the item in string format
                    h5file[path + '/' + str(key)] = str(item)

    with h5py.File(filename, 'w') as h5file:
        _save_dict_to_hdf5(h5file, '/', dic)
        h5file.close()

def hdf5_to_dict(h5_obj):

    dict_out = {}
    for key in h5_obj.keys():

        if isinstance(h5_obj[key], h5py.Group):
            dict_out[key] = hdf5_to_dict(h5_obj[key])  # Recursively convert subgroup to dictionary

        elif isinstance(h5_obj[key], h5py.Dataset):
            dict_out[key] = h5_obj[key][()]  # Convert dataset to NumPy array and store in dictionary
            # fix str to obj conversion bug
            if isinstance(dict_out[key], np.ndarray):
                if dict_out[key].dtype == 'O': dict_out[key] = dict_out[key].astype(str)
    
    return dict_out

def compute_QI(trials: np.ndarray, slice=(0,-1), mode=0):

        """
        Calculate response quality index as defined by
        Baden et al. over a matrix with shape (reps,time).
        """
        if mode == 0:
            
            # consider only valid trials (i.e with at least 2 values != 0)
            nz = nonzero(trials)
            if not nz:
                return 0
            
            a = np.var(trials[nz][:,slice[0]:slice[1]].mean(axis=0))
            b = np.mean(trials[nz][:,slice[0]:slice[1]].var(axis=1))
            return a / b
        
def filter(s, wn, ord=6, btype="low", analog=False, fs=None, mode="filtfilt"):

    """
    Apply scipy's filtfilt to signal s.
    """

    b, a = butter(ord, wn, btype, analog, fs=fs)

    if mode == "filtfilt":
        s_filtered = filtfilt(b, a, s)
    elif mode == "sosfiltfilt":
        s_filtered = sosfiltfilt(b, a, s)
    elif mode == "lfilter":
        s_filtered = lfilter(b, a, s)

    return s_filtered

def z_norm(s):

    """
    Compute z-score normalization on signal s
    """
        
    if not isinstance(s, np.ndarray):

        s = np.array(s)

    # if include_zeros:

    if s.ndim == 1:

        s_mean = np.mean(s)
        s_std = np.std(s)
        zscored =  (s - s_mean) / s_std

        # nz = nonzero(s)
        # zscored = s.copy()
        # s_mean = np.mean(s[nz])
        # s_std = np.std(s[nz])
        # zscored[nz] = (s[nz] - s_mean) / s_std
    
    else:
        # retrive rows with at least 2 values != 0
        nz = nonzero(s)
        zscored = s.copy()
        s_mean = np.mean(s[nz],axis=1)
        s_std = np.std(s[nz],axis=1)
        zscored[nz] = ((s[nz].T - s_mean) / s_std).T

    return zscored

def lin_norm(s, lb=0, ub=1):

    """
    Compute linear normalization between lb and ub on input signal s
    """
    return (ub - lb) * ((s - s.min()) / (s.max() - s.min())) + lb

def TSNE_embedding(data=None, **kwargs):

    if len(data) < 50:
        n_comp = len(data)
    else:
        n_comp = 50

    if kwargs:
        tsne_params = kwargs
    else:
        tsne_params = {
            "n_components": 2,
            "verbose": 1,
            "metric": "cosine",
            "early_exaggeration": 4,
            "perplexity": 15,
            "n_iter": 2000,
            "init": "pca",
            "angle": 0.1,
        }
    # run PCA
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(data)
    # run t-SNE
    tsne = TSNE(**tsne_params)

    transformed = tsne.fit_transform(transformed)

    return transformed

def MDR_embedding(data, ndims=3, smoothness=1, norm_mode=0):

    '''
    Mutual Distances Reduction:
    
    This is a dimensionality reduction algorithm that tries to project the trajectory of
    the state of a high-dimensional dynamical system by conserving the mutual distances between
    all the state variables at each time point.
     
    args:
    
    - data (array-like):
        data matrix in the shape n_features X n_variables
    - ndim (int):
        dimensionality of the embedding. For optimal 3d-visualization, use ndim=3
    - smoothness (float):
        weight of the smoothness penality used during the oprimization. 
    - norm_mode (int):
        can be 0, 1 or 2. specify a normalization method for step 4, se below

      '''

    ### step 1- compute cosine distance bettwen cells
    data_sparse = sparse.csr_matrix(data.T)
    distance = 1-cosine_similarity(data_sparse)

    ### step 2- project the distances in lower dimensional space using MDS
    mds_c = MDS(n_components=ndims, dissimilarity='precomputed',random_state=40)
    mds_c.__setattr__('normalized_stress', 'auto')

    dist_mds = mds_c.fit_transform(distance)

    ### step 3- normalize the distances and project on the unit sphere
    norms = np.linalg.norm(dist_mds, axis=1)
    coordinates = dist_mds / norms[:, np.newaxis]
    coordinates = coordinates / np.linalg.norm(coordinates, axis=1)[:, np.newaxis]

    ## compare with random
    # coordinates = np.random.uniform(-1,1,size=(data.shape[0],3))
    # coordinates = coordinates / np.linalg.norm(coordinates, axis=1)[:, np.newaxis]
    # print(coordinates.shape)

    ### step 4- normalize all the data between 0 and 2 (the values of each variable will be considered as distances to the fixed points)

    if norm_mode == 0:
        # type 1: normalize each variable across time independently
        data_norm = (data - np.min(data,axis=0)) / (np.max(data,axis=0) - np.min(data,axis=0))*2

    elif norm_mode == 1:
        # type 2: normalize each state vector independently. This should make each point in the embedding represent the istantaneous 
        # distances between the state variables
        data_norm = (data - np.min(data, axis=1)[:,np.newaxis]) / (np.max(data, axis=1)[:,np.newaxis] - np.min(data, axis=1)[:,np.newaxis])*2

    else:
        # type 3: normalize all the variables over time, together
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))*2

    ### step 5- represent the state of the system at each time as a 3d point inside the unit sphere. 
    #    we will minimize the difference between the distiances in the embedding (from fixed points and the state point)
    #    and the values of the variables, at each time point.

    # Objective function to minimize
    def stress_function(point, target_state, coords, penalty_weight, past_point=None):
        # calculate euclidean distances
        distances = np.sqrt(np.sum((coords-point)**2, axis=1))
        stress = np.sqrt(np.sum((distances - target_state)**2)/target_state.shape[0])

        if past_point is not None:
            # add penalty, defined as squared euclidean distance between previous and current point
            penalty = np.sum((point-past_point)**2)
            stress += penalty * penalty_weight

        return stress

    # Initial guess for the unknown point (center of the circle)
    initial_guess = np.zeros(ndims)
    # Optimize the position of the unknown point
    result = []
    optimized_distances = []
    rmses = []
    residuals = []

    for t,target_state in enumerate(tqdm(data_norm)):

        if t >0:
            args = (target_state, coordinates, smoothness, result[t-1].x)
        else:
            args = (target_state, coordinates, smoothness)

        res = minimize(stress_function, initial_guess, args=args, method='L-BFGS-B')

        # set the next initial guess with curretn guess
        initial_guess = res.x

        # save reuslts
        result.append(res)

        if not res.success:
            print('Warning: optimizer exited with negative status at iteration %d'%t)
        
        # save optimized distances
        opt_dist = np.sqrt(np.sum((coordinates-res.x)**2, axis=1))
        optimized_distances.append(opt_dist)

        # compute the  residuals and rmse
        residual = (target_state - opt_dist)**2
        residuals.append(residual)
        rmse = np.sqrt(np.sum(residual)/target_state.shape[0])
        rmses.append(rmse)
        
    result = np.array(result)
    optimized_distances = np.array(optimized_distances)
    residuals = np.array(residuals)

    # retrive coordinates
    projected_data = np.array([r.x for r in result])

    return projected_data


def k_means(data, n_clusters):

    # run Kmeans
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++").fit(data)

    return kmeans.labels_

def GMM(data, **kwargs):

    # run Gaussian Mixture Model
    gmm = BayesianGaussianMixture(**kwargs)
    gm_labels = gmm.fit_predict(data)

    return gm_labels

def find_optimal_kmeans_k(x):

    """
    Find the optimal number of clusters to use for k-means clustering on x.
    """

    # find optimal number of cluster for Kmeans
    Sum_of_squared_distances = []

    K = range(2, 8)

    for k in K:

        kmeans = KMeans(n_clusters=k, init="random").fit(x)

        Sum_of_squared_distances.append(kmeans.inertia_)

        labels = kmeans.labels_

    def _monoExp_(x, m, t, b):
        return m * np.exp(-t * x) + b

    x = np.arange(2, 8)

    p0 = (200, 0.1, 50)  # start with values near those we expect

    p, cv = curve_fit(_monoExp_, x, Sum_of_squared_distances, p0)

    m, t, b = p

    x_plot = np.arange(2, 8, 0.01)

    fitted_curve = _monoExp_(x_plot, m, t, b)

    # find the elbow point
    xx_t = np.gradient(x_plot)

    yy_t = np.gradient(fitted_curve)

    curvature_val = (
        np.abs(xx_t * fitted_curve - x_plot * yy_t)
        / (x_plot * x_plot + fitted_curve * fitted_curve) ** 1.5
    )

    dcurv = np.gradient(curvature_val)

    elbow = np.argmax(dcurv)

    plt.figure()
    plt.plot(x_plot, fitted_curve)
    plt.axvline(x_plot[elbow],0,fitted_curve.max(),color='r',linestyle='--',label='elbow')
    plt.xlabel('K')
    plt.ylabel('Sum_of_squared_distances')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    return round(x_plot[elbow])

def check_len_consistency(sequences, mode='trim', mean_len=10):

    """
    Utility function for correcting for length inconsistency in a list of 1-D iterables.

    - sequence (list of iterables):
        list of array-like elements that will be trimmed to the same (minimal) length.
    - mode (str):
        'trim': finds the size of the shortes iterable and trim the other iterables accordingly.
        'pad': finds the size of the longest iterable and mean-pad the other iterables accordingly.
    - mean_len (int):
        number of values used to calculate the mean used for padding if mode='pad'

    """

    # find minimal length

    lengths = [len(sequence) for sequence in sequences]

    if mode=='trim':
        
        min_len = np.min(lengths)

        # trim to minimal length

        sequences_new = []

        for seq in sequences:

            sequences_new.append(seq[:min_len])

    elif mode=='pad':

        max_len = np.max(lengths)

        # trim to minimal length

        sequences_new = []

        for seq in sequences:

            sequences_new.append(np.pad(seq,(len(seq)-max_len),mode='mean',stat_length=mean_len))

    return sequences_new

def nonzero(s):

    # retrive rows with at least 2 values != 0 in a 2d array
    x = np.where(s!=0)[0]
    counts = np.unique(x, return_counts=True)
    nzeros = [c1 for c1,c2 in zip(counts[0],counts[1]) if c2>2]
    return nzeros
    
def fit_oscillator(s, fs, plot_ps=False, freq_int=[0.2,0.2]):

    def optSinWrap(freq_osc=None):
        def optSin(x,freq_osc,phi):

            return np.sin(x*2*np.pi*freq_osc + phi)
        return optSin

    dt = 1 / fs
    #normalize signal
    sNorm = lin_norm(s,-1,1)
    # guess foundamental frequency of oscillation

    # OLD #
    # s_ps = np.abs(np.fft.fft(s))**2
    # freqs = np.fft.fftfreq(s.size,dt)
    # freq_osc = abs(freqs[np.argmax(s_ps[1:])+1])

    s_ps = np.abs(rfft(s))**2
    freqs = rfftfreq(s.size,dt)
    freq_osc = abs(freqs[np.argmax(s_ps[1:])+1])

    # le and ue are lower and upper edges of the frequencies spike which peak at freq_osc
    s_ps_diff = np.diff(s_ps[1:])
    le = freq_osc-freq_int[0] # freqs[np.argmax(s_ps_diff)+1]
    ue = freq_osc+freq_int[1] # freqs[np.argmin(s_ps_diff)+1]
    freq_range = np.array([le,ue])

    # idx = np.argsort(freqs) #useless

    if plot_ps:
        plt.figure()
        plt.title('{}Hz'.format(round(freq_osc,2)))
        plt.plot(freqs[:-1], np.diff(s_ps))
        # plt.plot(np.flip(freqs[:-1]), np.diff(np.flip(s_ps)))
        plt.plot(freqs, s_ps)
        plt.scatter(freq_osc,s_ps[1:].max(),c='r')
        plt.scatter(le,s_ps[np.argmax(s_ps_diff)+1],c='g')
        plt.scatter(ue,s_ps[np.argmin(s_ps_diff)+3],c='g')
        plt.show()

    # filter and normalize
    sfilt = filter(sNorm,freq_osc,btype='lp',fs=fs)
    sfiltNorm = lin_norm(sfilt,-1,1)

    # define time
    t = np.linspace(0,len(s)/fs,len(s))

    # optimize
    optSin=optSinWrap()
    popt = curve_fit(optSin,t,sfiltNorm,p0=[freq_osc,0],
                    bounds=([freq_osc-0.2,-np.pi],[freq_osc+0.2,np.pi]))[0]
    
    print('Freq peak:{}, Freq_fit:{}, Phase_opt:{}'.format(freq_osc,popt[0],popt[1]))

    return {'f':freq_osc,'fopt':popt[0],'phi':popt[1],'freq_range':freq_range},optSin(t,popt[0],popt[1])

def kill_freq(s, fs, f, freq_range, type='bs', lpcut=1.2):

    dt = 1/fs

    yf = rfft(s)
    W = rfftfreq(s.size, d=dt)
    cut_f = yf.copy()
    # kill freqs
    if type=='bs':
        cut_f[np.where((W>=freq_range[0])&(W<=freq_range[1]))] = 0 
        
    elif type=='bp':
        cut_f[np.where((W<=freq_range[0])|(W>=freq_range[1]))] = 0  

    cut_s = irfft(cut_f)
    # lowpass
    cut_s = filter(cut_s,lpcut,fs=fs)

    return cut_s

def deoscillate(x, x_train, fs, lpcut=1.2, norm=True, plot=False, freq_int=[0.2,0.2]):

    # estimate oscillation frequency from x_train
    # tosc = np.linspace(0,x_train.size/fs,x_train.size)
    if norm:
        x_train = lin_norm(x_train,-1,1)

    osc_params,osc_fit = fit_oscillator(x_train,fs,plot_ps=plot,freq_int=freq_int)

    x_filt = kill_freq(x,fs,osc_params['f'],osc_params['freq_range'])
    
    return x_filt