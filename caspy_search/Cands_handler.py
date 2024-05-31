import logging 
from sklearn.cluster import DBSCAN
import numpy as np
from . import constants as C

class Cands_handler:
    '''
    To be implemented
    '''
    def __init__(self, outname, clustering_eps=3):
        self.outname = outname
        self.header_inkeys = ['SNR', 'boxcar', 'DM', 'samp', 'ngroup']
        self.header_outkeys = self.header_inkeys + ['ncluster', 'boxcar_ms', 'DM_pccc', "time_s", "mjd_inf", "mjd_lower_edge"]
        self.db = DBSCAN(eps=clustering_eps, min_samples=1)
        self.open_outfile()

    def open_outfile(self):
        self.f = open(self.outname, 'w')
        for ik, key in enumerate(self.header_outkeys):
            if ik == 0:
                self.f.write(key)
            else:
                self.f.write("\t" + key)
        self.f.write("\n")
    
    def write_cands(self, cands):
        if len(cands) > 0:
            logging.debug(f"Writing {len(cands)} cands to file")
            for cand in cands:
                for ii, field in enumerate(cand):
                    if ii > 0:
                        self.f.write("\t")

                    if self.header_outkeys[ii] in ['mjd_inf', 'mjd_lower_edge']:
                        #This key is mjd, so do not round off the precision to 2 digits
                        self.f.write(f"{field:.11f}")
                    elif self.header_outkeys[ii] == 'time_s':
                        self.f.write(f"{field:.5f}")
                    else:
                        self.f.write(f"{field:.2f}")

                self.f.write("\n")
        else:
            logging.debug("No cands to write")

    def add_physical_units_columns(self, fbottom, df, nf, tsamp, mjd_start, cands):
        '''
        Convert DM, width and samp units to physical units
        Params
        ------
        fmin: Bottom edge freq of lowest channel (MHz)
        df: Channel Bandwidth (MHz)
        nf: Number of channels (int)
        tsamp: Sampling time (seconds)
        mjd_start: MJD start of the observation (float)
        cands : list of cands containing [SNR, boxcar, DM_samps, samp, ngroup, ncluster] values
        '''

        cands_arr = np.array(cands)
        ftop = ( fbottom + nf * df ) * C.MHZ_TO_GHZ
        fbottom *= C.MHZ_TO_GHZ
        final_cands = np.zeros((cands_arr.shape[0], len(self.header_outkeys)))
        #print("Shape of final cands is", final_cands.shape)
        n_in_keys = cands_arr.shape[1]
        final_cands[:, :n_in_keys] = cands_arr
        boxcar_ms = (final_cands[:, 1]+1) * tsamp * C.S_TO_MS       #+1 because boxcar 0 means 1 sample wide
        dm_pccc = final_cands[:, 2] * tsamp * C.S_TO_MS / C.DM_CONSTANT / (fbottom**-2 - ftop**-2)
        time_s = final_cands[:, 3] * tsamp
        delay_inf = C.DM_CONSTANT * dm_pccc * fbottom**-2 * C.MS_TO_S
        mjd_inf = mjd_start + (time_s - delay_inf) / C.S_IN_A_DAY 
        mjd_lower_edge = mjd_start + time_s / C.S_IN_A_DAY

        final_cands[:, n_in_keys] = boxcar_ms
        final_cands[:, n_in_keys + 1] = dm_pccc
        final_cands[:, n_in_keys + 2] = time_s
        final_cands[:, n_in_keys + 3] = mjd_inf
        final_cands[:, n_in_keys + 4] = mjd_lower_edge

        return final_cands

    def find_representative_cands(self, cands_arr, labels):
        best_cands = []
        uniq_labels = np.unique(labels)
        for label in uniq_labels:
            label_mask = labels == label
            this_label_cands = cands_arr[label_mask]
            ncands = len(this_label_cands)
            best_cand_idx = np.argmax(this_label_cands[:, 0])
            best_cand = list(this_label_cands[best_cand_idx])
            best_cand.append(ncands)
            best_cands.append(best_cand)
        return best_cands
        

    def cluster_cands(self, cands):
        cands_arr = np.array(cands)
        normalised_cands = np.zeros((len(cands), 3))
        normalised_cands[:, 0] = cands_arr[:, 1] / 4
        normalised_cands[:, 1] = cands_arr[:, 2] / 10
        normalised_cands[:, 2] = cands_arr[:, 3]

        clusters = self.db.fit(normalised_cands)

        repr_cands = self.find_representative_cands(cands_arr, clusters.labels_)
        return repr_cands



        
