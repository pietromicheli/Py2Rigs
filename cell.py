import numpy as np
from scipy.integrate import trapz

from Py2Rigs.utils import *


class Cell:

    """
    Base cell class 

    """

    def __init__(self, rec, id: int):

        self.id = id
        self.rec = rec
        self.label = 0 # usefull for pop analysis
        self.qi = 0
        self.responsive = 0
        self.analyzed_trials = {}
        self.params = rec.params
        self.sync = rec.sync
    
    def _compute_rmi_(self, a, b, mode_rmi="auc"):

        if mode_rmi == "auc":

            a_ = trapz(abs(a), dx=1)
            b_ = trapz(abs(b), dx=1)

        elif mode_rmi == "peak":

            a_ = np.max(abs(a))
            b_ = np.max(abs(b))

        rmi = (a_ - b_) / (a_ + b_)

        return rmi, a_, b_

    def _compute_snr_imp_(self, a, b, cutoff=0.2):

        """
        Extract noise and signal components from each signal,
        compute SNR and return ratio between SNRs
        """

        a_s = filter(a, cutoff, btype="low")
        a_n = filter(a, cutoff, btype="high")
        snr_a = abs(np.mean(a_s) / np.std(a_n))

        b_s = filter(b, cutoff, btype="low")
        b_n = filter(b, cutoff, btype="high")
        snr_b = abs(np.mean(b_s) / np.std(b_n))

        return snr_a / snr_b, snr_a, snr_b

    def is_responsive(self):

        """
        Asses responsiveness according to the QI value and the criteria specified in params
        """

        if self.params["resp_criteria"] == 1:
        
            qis_stims = []

            for stim in self.analyzed_trials:

                qis_trials = []

                for trial_name in self.analyzed_trials[stim]:

                    qis_trials.append(self.analyzed_trials[stim][trial_name]["QI"])

                # for each stimulus, consider only the highest QI calculated
                # over all the different trial types (i.e. Ipsi, Both, Contra)
                qis_stims.append(max(qis_trials))

            if self.params["qi_metrics"]==0:
                responsive = all([qi >= self.params["qi_threshold"] for qi in qis_stims])

            else:
                responsive = all([qi <= 0.05 for qi in qis_stims])

        else:

            if self.params["qi_metrics"]==0:
                responsive = (self.qi >= self.params["qi_threshold"])

            else:
                responsive = (self.qi <= 0.05)


        self.responsive = responsive

        return responsive

    def calculate_modulation(self, stim, trial_name_1, trial_name_2, rtype='norm', mode="rmi", slice=(0,-1), **kwargs):

        """
        Calculate Response Modulation Index on averaged responses to
        trial_type_1 vs trial_type_2 during stimulus stim.

        - stim: str
            stimulus name
        - trial_type_1: str
            first trial_type name (i.e "BOTH","CONTRA","IPSI")
        - trial_type_2: str
            second trial_type name (i.e "BOTH","CONTRA","IPSI"),
            it is supposed to be different from trial_type_1.
        - slice: tuple
            slice indexes to use for extracting the portion of the traces that will
            be used for calculating the modulation.
        """

        if not self.analyzed_trials:

            self.analyze()

        average_resp_1 = self.analyzed_trials[stim][trial_name_1]["%s_avg"%rtype][slice[0]:slice[1]]
        average_resp_2 = self.analyzed_trials[stim][trial_name_2]["%s_avg"%rtype][slice[0]:slice[1]]


        if mode == "rmi":

            mod = self._compute_rmi_(average_resp_1, average_resp_2, **kwargs)

        elif mode == "snr":

            mod = self._compute_snr_imp_(average_resp_1, average_resp_2, **kwargs)

        return mod

