import numpy as np
from fdmt import Fdmt

class Dedisperser:
    def __init__(self, fmin:float, df: float, nf: int, max_dm: int, nt:int):
        self.fmin = fmin
        self.df = df
        self.nf = nf
        self.max_dm = max_dm
        self.nt = nt

        self.fdmt = Fdmt(fmin, df, nf, max_dm, nt)
        self.summed_dmt_out = np.zeros((max_dm, max_dm + nt))
    
    def get_single_block_DMT(self, block):
        dmt_out = self.fdmt(block)
        return dmt_out

    def overlap_and_sum(self, dmt):
        ndm, ndm_plus_nt = dmt.shape
        nt = ndm_plus_nt - ndm
        for idm in range(ndm):
            for it in range(ndm_plus_nt):
                if it + nt < ndm_plus_nt:
                    self.summed_dmt_out[idm, it] = dmt[idm, it] + self.summed_dmt_out[idm, it + nt]
                else:
                    self.summed_dmt_out[idm, it] = dmt[idm, it]

    def get_full_DMT(self, block):
        dmt_single = self.get_single_block_DMT(block)
        self.overlap_and_sum(dmt_single)

        return self.summed_dmt_out[:, :self.nt] / np.sqrt(self.nf)




    