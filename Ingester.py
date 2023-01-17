from sigpyproc.readers import FilReader
import numpy as np
import logging

class LoadFilterbank(FilReader):
    def __init__(self, filname:str):
        super().__init__(filname)
        self.ftop = np.max(self.header.chan_freqs) + np.abs(self.header.foff) / 2
        self.fbottom = np.min(self.header.chan_freqs) - np.abs(self.header.foff) / 2
        self.df = np.abs(self.header.foff)
        self.nchans = self.header.nchans
        self.tot_samples = self.header.nsamples

    def yield_block(self, nt:int = 256, start:int = 0, nsamps : int = None, skipback: int=0 ):
        for iblock, _, block in self.read_plan(nt, start, nsamps, skipback, verbose = False):
            if block.size != self.nchans * nt:
                nt_this_block = block.size // self.nchans
                logging.info(f"Block {iblock} only has {nt_this_block} samps, yielding a smaller block")
            else:
                nt_this_block = nt

            yield block.reshape(nt_this_block, -1).T

    def get_block(self, start:int = 0, nsamps: int = None):
        if nsamps is None:
            nsamps = self.tot_samples
        return self.read_block(start, nsamps)

