import numpy as np
import logging
from numba import jit#, int32, float64
import matplotlib.pyplot as plt
#from numba.experimental import jitclass

logging.getLogger('numba').setLevel(logging.WARNING)
'''
spec = [
    ('ndm', int32),
    ('nbox', int32),
    ('nt', int32),
    ('boxcar_history', float64[:, :]),
    ('dmt_out', float64[:, :]),
    ('threshold', int32),
    ('dm_boxcar_norm_factors', float64[:, :]),
    ('iblock', int32)
]
'''
@jit(nopython=True)
def do_only_boxcar(dmt_out, ndm, nbox, nt, boxcar_history):

    summed = np.zeros((ndm, nt, nbox), dtype=np.float64)
    for idm in range(ndm):
        for it in range(nt):
            bcsum = 0
            for ibox in range(nbox):
                if it >= ibox:
                    inv = dmt_out[idm, it - ibox]
                else:
                    inv = boxcar_history[idm, it -ibox]

                bcsum = bcsum + inv
                summed[idm, it, ibox] = bcsum
    
    return summed
    


@jit(nopython=True)
def do_boxcar_and_threshold(dmt_out, threshold, dm_boxcar_norm_factors, iblock, ndm, nbox, nt, boxcar_history, keep_last_boxcar):
    candidates = np.zeros((ndm * nt, 5), dtype=np.float64) 
    cands_recorded = 0
    cands_ignored = 0
    #plt.figure()
    #logging.debug(f"Threshold is set as: {threshold}")
    for idm in range(ndm):
        #snrs = []
        for it in range(nt):
            bcsum = 0
            best_cand = np.zeros(5, dtype=np.float64)
            best_snr = None
            ngroup = 1
            for ibox in range(nbox):
                if it >= ibox:
                    inv = dmt_out[idm, it - ibox]
                else:
                    inv = boxcar_history[idm, it - ibox]
                bcsum = bcsum + inv
                #snr = bcsum / np.sqrt(ibox + 1)
                snr = bcsum * dm_boxcar_norm_factors[idm, ibox]
                
                
                if snr >= threshold:
                    if best_snr is None or snr > best_snr:
                        best_cand[0] = snr
                        best_cand[1] = ibox
                        best_cand[2] = idm
                        best_cand[3] = it + iblock * nt
                        best_snr = snr
                        #print(f"The best snr for idm {idm}, it {it} and ibox {ibox} before normalising was: {snr / dm_boxcar_norm_factors[idm, ibox]}, {snr}, {dm_boxcar_norm_factors[idm, ibox]}")
                    else:
                        ngroup = ngroup + 1
                    
            if best_cand[0] > 0:
                if best_cand[1] == (nbox-1) and not keep_last_boxcar:
                    cands_ignored += 1
                    continue
                best_cand[4] = ngroup
                candidates[cands_recorded] = best_cand
                cands_recorded = cands_recorded + 1
                #snrs.append(snr)
        #print(f"len(snrs) = {len(snrs)}, candidates_recorded = {cands_recorded}")
        #plt.plot(snrs, label=f"idm = {idm}")
    #plt.show()
    print(f"Boxcar and threshold found {cands_recorded} cands, and ignored {cands_ignored} cands, ndm * nt = {ndm * nt}")
    return candidates[:cands_recorded]


#@jitclass(spec)
class Boxcar_and_threshold:
    def __init__(self, nt, boxcar_history, keep_last_boxcar):
        ndm, nbox = boxcar_history.shape
        self.ndm = ndm
        self.nbox = nbox
        self.nt = nt
        self.boxcar_history = boxcar_history
        self.keep_last_boxcar = keep_last_boxcar


    def boxcar_and_threshold(self, dmt_out, threshold, dm_boxcar_norm_factors, iblock):
        candidates = do_boxcar_and_threshold(dmt_out, threshold, dm_boxcar_norm_factors, iblock, self.ndm, self.nbox, self.nt, self.boxcar_history, self.keep_last_boxcar)
        self.boxcar_history = dmt_out[:, -self.nbox:]
        return candidates

    def run_pure_boxcar(self, dmt_out):
        boxed_out = do_only_boxcar(dmt_out, self.ndm, self.nbox, self.nt, self.boxcar_history)
        return boxed_out
        

class Boxcarer:
    def __init__(self, nf:int, max_boxcar_width:int = 32):
        self.max_boxcar_width = max_boxcar_width
        self.boxcar_history = np.zeros((nf, max_boxcar_width))

    def convolve(data, boxcar):
        if boxcar == 1:
            return data
        ts_conv = np.convolve(data, np.ones(boxcar), mode='valid')
        return ts_conv

    def run_boxcar(self, data):
        new_data = np.hstack((self.boxcar_history, data))
        self.boxcar_history = data[:, -self.max_boxcar_width:]
        boxout = []
        for ibox in range(1, self.max_boxcar_width + 1):
            boxout.append(self.convolve(new_data, ibox))

        return np.array(boxout)

    
class Thersholder:
    def __init__(self, threshold = 8):
        self.threshold = threshold

    def threshold(self, cube):
        locs = np.argswhere(cube > self.thershold)
        return locs, cube[locs]
