import logging 
from sklearn.cluster import DBSCAN
import numpy as np

class Cands_handler:
    '''
    To be implemented
    '''
    def __init__(self, outname, clustering_eps=3):
        self.outname = outname
        self.header_inkeys = ['SNR', 'boxcar', 'DM', 'samp', 'ngroup']
        self.header_outkeys = self.header_inkeys + ['ncluster', 'boxcar_ms', 'DM_pccc', "time_s"]
        self.db = DBSCAN(eps=clustering_eps, min_samples=1)
        self.open_outfile()

    def open_outfile(self):
        self.f = open(self.outname, 'w')
        for key in self.header_outkeys:
            self.f.write(key + "\t")
        self.f.write("\n")
    
    def write_cands(self, cands):
        if len(cands) > 0:
            logging.debug(f"Writing {len(cands)} cands to file")
            for cand in cands:
                for field in cand:
                    self.f.write(f"{field:.2f}\t")
                self.f.write("\n")
        else:
            logging.debug("No cands to write")

    def add_physical_units_columns(self, fbottom, df, nf, tsamp, cands):
        '''
        Convert DM, width and samp units to physical units
        Params
        ------
        fmin: Bottom edge freq of lowest channel (MHz)
        df: Channel Bandwidth (MHz)
        nf: Number of channels (int)
        tsamp: Sampling time (seconds)
        '''

        cands_arr = np.array(cands)
        ftop = ( fbottom + nf * df ) * 1e-3     #Converting into GHz
        fbottom *= 1e-3         #Converting into GHz
        final_cands = np.zeros((cands_arr.shape[0], cands_arr.shape[1] + 3))
        print("Shape of final cands is", final_cands.shape)
        n_in_keys = cands_arr.shape[1]
        final_cands[:, :n_in_keys] = cands_arr
        final_cands[:, n_in_keys] = final_cands[:, 1] * tsamp * 1e3
        final_cands[:, n_in_keys + 1] = final_cands[:, 2] * tsamp * 1e3 / 4.15 / (fbottom**-2 - ftop**-2)
        final_cands[:, n_in_keys + 2] = final_cands[:, 3] * tsamp
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



        