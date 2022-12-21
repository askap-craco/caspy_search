import numpy as np
import logging
from iqrm import iqrm_mask

def get_mad(median_normed_data):
    return np.median(np.abs(median_normed_data), axis=1)

def get_mad_std(median_normed_data):
    return 1.4286 * get_mad(median_normed_data)

class Normalise:
    
    def mad_norm(self, data, mean = 0, std = 1):
        if data.ndim != 2:
            raise ValueError("Can only handle 2-D arrays in FT order at the moment")

        medians = np.median(data, axis=1)   
        median_normed_data = data - medians[:, None]
        
        old_std = get_mad_std(median_normed_data)
        non_zero_std_mask = old_std != 0
        normed_data = np.zeros_like(median_normed_data)
        normed_data[non_zero_std_mask, :] = median_normed_data[non_zero_std_mask, :] / old_std[non_zero_std_mask, None] * std + mean

        return normed_data

    def std_norm(self, data, mean=0, std = 1):
        if data.ndim !=2 :
            raise ValueError("Can only handle 2-D arrays in FT order at the moment")

        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1)

        non_zero_std_mask = stds !=0
        normed_data = np.zeros_like(data)
        normed_data[non_zero_std_mask, :] = (data[non_zero_std_mask, :] - means[non_zero_std_mask, None]) / stds[non_zero_std_mask, None] * std + mean
        return normed_data


class RFI_mitigator:
    def __init__(self, cleaning_chunk = 64, threshold = 3):
        self.chunk = cleaning_chunk 
        self.threshold = threshold
        logging.debug(f"Setting chunk size = {self.chunk}")
    
    def get_chunk(self, block):
        nf, nt = block.shape
        if nt <= self.chunk:
            logging.debug("Block can only fit one chunk -- yielding the full block as a chunk")
            yield block
        else:
            nchunks = nt // self.chunk
            if nt %self.chunk !=0:
                nchunks += 1
            logging.debug(f"Breaking the block into {nchunks} chunks")

            for ichunk in range(nchunks):
                start = ichunk * self.chunk
                end = (ichunk + 1) * self.chunk
                if end > nt:
                    end = nt
                yield block[:, start:end]

    def get_mask(self, ts):
        mask, votes = iqrm_mask(ts, radius = len(ts) / 10, threshold = self.threshold)
        return mask

    def clean_block(self, block):
        for ichunk, chunk in enumerate(self.get_chunk(block)):
            chan_mask = self.get_mask(chunk.std(axis=1))
            chunk[chan_mask, :] = 0
            time_mask = self.get_mask(chunk.std(axis=0))
            chunk[:,time_mask] = 0

            if ichunk ==0:
                cleaned_block = chunk.copy()
            else:
                cleaned_block = np.hstack([cleaned_block, chunk])
        
        logging.debug("Finished cleaning the block")

        return cleaned_block


        