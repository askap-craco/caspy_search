from sigpyproc.readers import FilReader
import numpy as np
import logging

class LoadFilterbank(FilReader):
    def __init__(self, filname:str):
        super().__init__(filname)
        self.ftop = np.max(self.header.chan_freqs) + np.abs(self.header.foff) / 2
        self.fbottom = np.min(self.header.chan_freqs) - np.abs(self.header.foff) / 2
        self.df = np.abs(self.header.foff)
        self.needs_flipping = self.header.foff < 0
        self.nchans = self.header.nchans
        self.tsamp = self.header.tsamp
        self.tot_samples = self.header.nsamples

    def yield_block(self, nt:int = 256, start:int = 0, nsamps : int = None, skipback: int=0 , flip_band:bool=False):
        for iblock, _, block in self.read_plan(nt, start, nsamps, skipback ):
            if block.size != self.nchans * nt:
                nt_this_block = block.size // self.nchans
                logging.info(f"Block {iblock} only has {nt_this_block} samps, ending the yield loop now")
                break
            else:
                nt_this_block = nt

            outblock = block.reshape(nt_this_block, -1).T
            if flip_band:
                outblock = outblock[::-1, :]
                logging.debug("Flipping the band because it was asked for")
            if self.needs_flipping:
                logging.debug("The channel bandwidth is positive --- flipping the data")
                yield outblock[::-1, :]
            else:
                yield outblock

    def get_block(self, start:int = 0, nsamps: int = None):
        if nsamps is None:
            nsamps = self.tot_samples
        return self.read_block(start, nsamps)

class LoadNumpy:
    def __init__(self, filename:str, fbottom:float=1152.5, df:float=1.0, tsamp:float=0.001):
        self.filename = filename
        self.data = np.load(self.filename)
        if self.data.ndim != 2:
            raise TypeError("The datafile does not contain a 2-D array")
        self.nchans = self.data.shape[0]
        self.fbottom = fbottom
        self.df = df
        self.tot_samples = self.data.shape[1]
        if self.df > 0:
            self.data = self.data[::-1, :]

    def yield_block(self, nt:int=256, start:int = 0, nsamps:int = None, skipback:int =0):
        last_samp_read = start
        iblock = 0
        if nsamps is None:
            nsamps = self.tot_samples - start

        stop_samp = nsamps // nt * nt + start
        while True:
            if last_samp_read >= start + nsamps or last_samp_read >= self.tot_samples:
                logging.info("Reached the end of file.. exiting the yielding while loop")
                break
            start_samp = last_samp_read
            if iblock > 0:
                start_samp = last_samp_read - skipback

            if start_samp + nt > stop_samp:
                logging.info("The last block would have less than nt samples, so ending the yield loop now")
                break

            end_samp = start_samp + nt
            block = self.data[:, start_samp:end_samp]

            logging.debug(f"Yielding block {iblock} with shape = {block.shape}")
            last_samp_read = end_samp
            iblock += 1

            yield block
            
    def get_block(self, start:int=0, nsamps:int=None):
        if start > self.tot_samples:
            raise ValueError("start > tot_nsamps")
        if nsamps < 0:
            raise ValueError("nsamps cannot be < 0, say None for all samps")
        
        if nsamps is None:
            return self.data[:, start:]
        else:
            return self.data[:, start:start + nsamps]
    



        
