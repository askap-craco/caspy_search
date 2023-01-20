from Ingester import LoadFilterbank as F
from Ingester import LoadNumpy as LN
from Preprocessor import Normalise, RFI_mitigator
from Dedisperser import Dedisperser
from Boxcarer import Boxcar_and_threshold
from Cands_handler import Cands_handler
import numpy as np
import argparse, os, time
import logging
import matplotlib.pyplot as plt


def run_search(fil_name, nt, max_dm, max_boxcar, threshold, candfile):
    if fil_name.endswith("fil"):
        f = F(fil_name)
    elif fil_name.endswith("npy"):
        f = LN(fil_name)
    else:
        raise ValueError("Invalid data type provided")
    tot_nblocks = f.tot_samples // nt  + 1 if f.tot_samples %nt >0 else f.tot_samples //nt
    norm = Normalise()
    rfi_m = RFI_mitigator(cleaning_chunk=128)
    dd = Dedisperser(fmin = f.fbottom, df = f.df, nf = f.nchans, max_dm=max_dm, nt = nt)
    bt = Boxcar_and_threshold(nt = nt, boxcar_history=np.zeros((max_dm, max_boxcar)))
    dm_boxcar_norm_factors = 1 / dd.get_rms_normalising_factor(max_boxcar)
    #print("dm_boxcar_norm_factors are:", dm_boxcar_norm_factors)
    ch = Cands_handler(outname = candfile)
    all_cands = []
    max_cands_per_block = 1e6
    for iblock, block in enumerate(f.yield_block(nt)):
        start = time.time()
        logging.info(f"Processing block {iblock}")
        if args.plot:
            plt.figure()
            plt.imshow(block, aspect='auto', interpolation='None')
            plt.title("Raw block")

        logging.debug(f"Normalising the block")
        normed_block = norm.mad_norm(block)
        norm_time = time.time()
        if args.plot:
            plt.figure()
            plt.imshow(normed_block, aspect='auto', interpolation='None')
            plt.title("Normed block")

        logging.debug("Cleaning the block")
        cleaned_block = rfi_m.clean_block(normed_block)
        cleaning_time = time.time()
        if args.plot:
            plt.figure()
            plt.imshow(cleaned_block, aspect='auto', interpolation='None')
            plt.title("Cleaned block")

        logging.debug("Dedispersing the block")
        dd_block = dd.get_full_DMT(cleaned_block)
        disp_time = time.time()
        if args.plot:
            plt.figure()
            plt.imshow(dd_block, aspect='auto', interpolation='None')

        logging.debug("Running boxcar and threshold")
        cands = bt.boxcar_and_threshold(dd_block, threshold=threshold, dm_boxcar_norm_factors=dm_boxcar_norm_factors, iblock = iblock)
        bt_time = time.time()
        ncands = len(cands)
        logging.debug(f"Got {ncands} cands in block {iblock}.")
        if ncands > max_cands_per_block:
            logging.info("Got too many cands in this block, dropping all of them")
            continue
        if ncands > 0:
            all_cands.extend(cands)
        if (iblock > 0 and iblock % 5 == 0 and len(all_cands) > 0) or (iblock + 1 == tot_nblocks):
            logging.debug("Clustering the candidates now")
            repr_cands = ch.cluster_cands(all_cands)
            logging.debug(f"Writing the clustered cands to {candfile}")
            ch.write_cands(repr_cands)
            all_cands = []
        end = time.time()
        logging.debug(f"It took a total of {end - start}s")
        logging.debug(f"The breakdown of times is as follows - start = {start - start}, norm = {norm_time - start}, cleaning = {cleaning_time - norm_time}, dmt_time = {disp_time - cleaning_time}, boxcar_and_thresh = {bt_time - disp_time} ")

        if args.plot:
            plt.show()

    logging.info("Closing cand file")
    ch.f.close()

def set_up_logging(log_level):
    if log_level.upper() not in ['CRITICAL', 'INFO', 'DEBUG']:
        raise ValueError("Invalid logging level specified: ", log_level)
    logging_level = logging.__getattribute__(log_level.upper())
    logging.basicConfig(level=logging_level, format="%(asctime)s, %(levelname)s: %(message)s")

def main(args):
    set_up_logging(args.log_level)
    candname = args.C
    if args.C is None:
        basename = args.f
        candname = "".join(basename.split(".")[:-1]) + ".cand"
    run_search(args.f, args.nt, args.max_dm, args.max_boxcar, args.T, candname)

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("-f", type=str, help="Filterbank to process", required=True)
    a.add_argument("-nt", type=int, help="No. of samples in each block to process (def: 256)", default=256)
    a.add_argument("-max_dm", type=int, help="Max DM (in sample units) to search upto (def = 1024)", default=1024)
    a.add_argument("-max_boxcar", type=int, help="Max boxcar width (in samps) to search upto (def = 32)", default=32)
    a.add_argument("-T", type=float, help="S/N threshold to get candidates (def = 10)", default=10)
    a.add_argument("-C", type=str, help="Name of file to write cands to (def = <filname>.cands", default=None)
    a.add_argument("-log_level", type=str, help="Logging level - [CRITICAL/INFO/DEBUG] (def = DEBUG)", default="DEBUG")
    a.add_argument("-plot", action='store_true', help="Plot the different stages of processing for each block?", default=False)

    args = a.parse_args()
    main(args)