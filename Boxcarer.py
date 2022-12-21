import numpy as np
import logging

class Boxcar_and_threshold:
    def __init__(self, nt, boxcar_history):
        ndm, nbox = boxcar_history.shape
        self.ndm = ndm
        self.nbox = nbox
        self.nt = nt
        self.boxcar_history = boxcar_history

    def boxcar_and_threshold(self, dmt_out, threshold=10):
        candidates = []

        for idm in range(self.ndm):
            for it in range(self.nt):

                bcsum = 0
                best_cand = None
                best_snr = None
                ngroup = 0

                for ibox in range(self.nbox):

                    if it >= ibox:
                        inv = dmt_out[idm, it - ibox]
                    else:
                        inv = self.boxcar_history[idm, -ibox]

                    bcsum += inv
                    snr = bcsum / np.sqrt(ibox + 1)

                    if snr >= threshold:

                        if best_snr is None or snr > best_snr:
                            best_cand = [snr, ibox, idm, it]
                            best_snr = snr
                        else:
                            ngroup += 1
                        
                if best_cand is not None:
                    best_cand.append(ngroup)
                    candidates.append(best_cand)

        self.boxcar_history = dmt_out[:, -self.nbox:]

        return candidates

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
