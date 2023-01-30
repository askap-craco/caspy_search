import numpy as np
from fdmt import Fdmt
from numba import jit
import logging

@jit(nopython=True)
def fast_overlap_and_sum(dmt, summed_dmt_out):
    ndm, ndm_plus_nt = dmt.shape
    nt = ndm_plus_nt - ndm
    for idm in range(ndm):
        for it in range(ndm_plus_nt):
            if it + nt < ndm_plus_nt:
                summed_dmt_out[idm, it] = dmt[idm, it] + summed_dmt_out[idm, it + nt]
            else:
                summed_dmt_out[idm, it] = dmt[idm, it]  
    
    return summed_dmt_out 

class Dedisperser:
    def __init__(self, fmin:float, df: float, nf: int, max_dm: int, nt:int):
        self.fmin = fmin
        self.df = df
        self.nf = nf
        self.max_dm = max_dm
        self.nt = nt

        self.fdmt = Fdmt(fmin, df, nf, max_dm, nt)
        self.summed_dmt_out = np.zeros((max_dm, max_dm + nt))

        self.nsamps_summed = self.get_nsamps_summed()
        logging.info(f"Setting up the Dedisperser class with fmin = {self.fmin}, df = {self.df}, nf = {self.nf}, max_dm = {self.max_dm}, nt = {self.nt}")

    def get_nsamps_summed(self):
        mock_fdmt = Fdmt(self.fmin, self.df, self.nf, self.max_dm, self.max_dm + 1)
        mock_data = np.ones((self.nf, self.max_dm + 1))
        mock_dmt = mock_fdmt(mock_data)
        return mock_dmt[:, self.max_dm]

    def get_rms_normalising_factor(self, max_boxcar):
        eff_var = np.zeros([self.max_dm, max_boxcar])
        for idm in range(self.max_dm):
            for ibox in range(max_boxcar):
                eff_var[idm, ibox] = self.fdmt.get_eff_var_recursive(idm, ibox+1)
        return eff_var**0.5

    def get_single_block_DMT(self, block):
        dmt_out = self.fdmt(block)
        return dmt_out

    def overlap_and_sum(self, dmt):
        self.summed_dmt_out = fast_overlap_and_sum(dmt, self.summed_dmt_out)

    def get_full_DMT(self, block):
        dmt_single = self.get_single_block_DMT(block)
        self.overlap_and_sum(dmt_single)

        return self.summed_dmt_out #/ np.sqrt(self.nsamps_summed)[:, None]




    