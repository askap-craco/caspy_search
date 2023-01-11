from Ingester import LoadFilterbank as F
from Preprocessor import Normalise, RFI_mitigator
from Dedisperser import Dedisperser
from Boxcarer import Boxcar_and_threshold
from Cands_handler import Cands_handler
import numpy as np
import argparse, os
import logging
import matplotlib.pyplot as plt


def run_search(fil_name, nt, max_dm, max_boxcar, threshold, candfile):
    f = F(fil_name)
    norm = Normalise()
    rfi_m = RFI_mitigator(cleaning_chunk=128)
    dd = Dedisperser(fmin = f.fbottom, df = f.df, nf = f.nchans, max_dm=max_dm, nt = nt)
    bt = Boxcar_and_threshold(nt = nt, boxcar_history=np.zeros((max_dm, max_boxcar)))
    dm_boxcar_norm_factors = dd.get_rms_normalising_factor(max_boxcar)
    ch = Cands_handler(outname = candfile)

    for iblock, block in enumerate(f.yield_block(nt)):
        logging.info(f"Processing block {iblock}")
        #plt.figure()
        #plt.imshow(block, aspect='auto', interpolation='None')
        #plt.title("Raw block")

        logging.debug(f"Normalising the block")
        normed_block = norm.mad_norm(block)
        #plt.figure()
        #plt.imshow(normed_block, aspect='auto', interpolation='None')
        #plt.title("Normed block")

        logging.debug("Cleaning the block")
        cleaned_block = rfi_m.clean_block(normed_block)
        #plt.figure()
        #plt.imshow(cleaned_block, aspect='auto', interpolation='None')
        #plt.title("Cleaned block")

        logging.debug("Dedispersing the block")
        dd_block = dd.get_full_DMT(cleaned_block)
        #plt.figure()
        #plt.imshow(dd_block, aspect='auto', interpolation='None')

        logging.debug("Running boxcar and threshold")
        cands = bt.boxcar_and_threshold(dd_block, threshold=threshold, dm_boxcar_norm_factors=dm_boxcar_norm_factors, iblock = iblock)

        logging.debug(f"Got {len(cands)} cands in block {iblock}.")
        logging.debug(f"Writing the cands to {candfile}")
        ch.write_cands(cands)
        #plt.show()

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
        basename = os.path.basename(args.f)
        candname = basename.split(".")[0] + ".cand"
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

    args = a.parse_args()
    main(args)