from caspy_search.Ingester import LoadFilterbank as F
from caspy_search.Ingester import LoadNumpy as LN
from caspy_search.Preprocessor import Normalise, RFI_mitigator
from caspy_search.Dedisperser import Dedisperser
from caspy_search.Boxcarer import Boxcar_and_threshold
from caspy_search.Cands_handler import Cands_handler
import numpy as np
import argparse, os, time, sys
import logging
import matplotlib.pyplot as plt


def run_search(fil_name, nt, max_dm, max_boxcar, threshold, candfile, args):
    if fil_name.endswith("fil"):
        f = F(fil_name)
    elif fil_name.endswith("npy"):
        f = LN(fil_name)
    else:
        raise ValueError("Invalid data type provided")

    start, nsamps = 0, f.tot_samples
    if args.seek is not None:
        start = int(args.seek // f.tsamp)
    elif args.seek_samps is not None:
        start = args.seek_samps
    if args.dur is not None:
        nsamps = int(args.dur // f.tsamp)
    elif args.ns is not None:
        nsamps = args.ns

    tot_nblocks = f.tot_samples // nt  + 1 if f.tot_samples %nt >0 else f.tot_samples //nt
    norm = Normalise(care_about_zeros=True)
    rfi_m = RFI_mitigator(cleaning_chunk=128)
    dd = Dedisperser(fmin = f.fbottom, df = f.df, nf = f.nchans, max_dm=max_dm, nt = nt)
    bt = Boxcar_and_threshold(nt = nt, boxcar_history=np.zeros((max_dm, max_boxcar)), keep_last_boxcar=args.keep_last_boxcar)
    dm_boxcar_norm_factors = 1 / dd.get_rms_normalising_factor(max_boxcar)
    if args.plot:
        plt.figure()
        #print("dm_boxcar_norm_factors are:", dm_boxcar_norm_factors)
        plt.imshow(dm_boxcar_norm_factors, aspect='auto', interpolation='None')
        plt.xlabel("Boxcar")
        plt.ylabel("iDM")
        plt.title("dm_boxcar_normalising_factors")
        plt.show()
    ch = Cands_handler(outname = candfile, clustering_eps = args.cl_eps)
    all_cands = []
    max_cands_per_block = max_dm * nt // 2 
    for iblock, block in enumerate(f.yield_block(nt, start, nsamps, flip_band = args.flip_band)):
        start_time = time.time()
        logging.info(f"Processing block {iblock}")
        if args.plot:
            plt.figure()
            plt.imshow(block, aspect='auto', interpolation='None')
            plt.title("Raw block")

        logging.debug(f"Normalising the block")
        normed_block = norm.proper_norm(block)
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

            boxed_out = bt.run_pure_boxcar(dd_block)
            tndm, tnt, tnbox = boxed_out.shape
            for ibox in range(tnbox):
                plt.figure()
                plt.imshow(boxed_out[:, :, ibox] * dm_boxcar_norm_factors[:, ibox].reshape(tndm, -1), aspect='auto', interpolation="None")
                plt.title(f"ibox = {ibox}")

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
        if (len(all_cands) > 0) and ((iblock > 0 and iblock % args.clf == 0) or (iblock + 1 == tot_nblocks) or (iblock + 2 == tot_nblocks)):
            logging.debug(f"Clustering the {len(all_cands)} candidates now")
            repr_cands = ch.cluster_cands(all_cands)
            final_cands = ch.add_physical_units_columns(cands=repr_cands, fbottom=f.fbottom, df = f.df, nf = f.nchans, tsamp=f.tsamp, mjd_start = f.header.tstart)
            logging.debug(f"Writing {len(final_cands)} clustered cands to {candfile}")
            ch.write_cands(final_cands)
            all_cands = []
        end = time.time()
        logging.info(f"It took a total of {end - start_time}s")
        logging.info(f"The breakdown of times is as follows - norm = {norm_time - start_time}, cleaning = {cleaning_time - norm_time}, dmt_time = {disp_time - cleaning_time}, boxcar_and_thresh = {bt_time - disp_time} ")

        if args.plot:
            plt.show(block=True)
            _ = input()
            plt.close('all')

    ch.f.write("# This file contains the output of the following command: " + " ".join(sys.argv[:]) + "\n")
    logging.info("Closing cand file")

    ch.f.close()

def set_up_logging(log_level):
    if log_level.upper() not in ['CRITICAL', 'INFO', 'DEBUG']:
        raise ValueError("Invalid logging level specified: ", log_level)
    logging_level = logging.__getattribute__(log_level.upper())
    logging.basicConfig(level=logging_level, format="%(asctime)s, %(levelname)s: %(message)s")


def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-f", type=str, help="Filterbank to process", required=True)
    a.add_argument("-nt", type=int, help="No. of samples in each block to process (def: 256)", default=256)
    a.add_argument("-max_dm", type=int, help="Max DM (in sample units) to search upto (def = 1024)", default=1024)
    a.add_argument("-max_boxcar", type=int, help="Max boxcar width (in samps) to search upto (def = 32)", default=32)
    a.add_argument("-T", type=float, help="S/N threshold to get candidates (def = 10)", default=10)
    a.add_argument("-C", type=str, help="Name of file to write cands to (def = <filname>.cands", default=None)
    a.add_argument("-log_level", type=str, help="Logging level - [CRITICAL/INFO/DEBUG] (def = DEBUG)", default="DEBUG")
    a.add_argument("-clf", type=int, help="How many blocks to accumulate before clustering (def = 5)", default=5)
    a.add_argument("-plot", action='store_true', help="Plot the different stages of processing for each block?", default=False)
    a.add_argument("-flip_band", action='store_true', help="Flip the data along freq axis (def:False)", default=False)
    a.add_argument("-keep_last_boxcar", action='store_true', help="Dont discard cands in the highest boxcar trial", default=False)
    a.add_argument("-cl_eps", type=float, help="Clustering eps (def=2.5)", default=2.5)
    g1 = a.add_mutually_exclusive_group()
    g1.add_argument("-seek", type=float, help="Seek x seconds into the file (def = 0)", default=None)
    g1.add_argument("-seek_samps", type=int, help="Seek x samps into the file (def =0)", default=None)
    g2 = a.add_mutually_exclusive_group()
    g2.add_argument("-dur", type=float, help="Process x seconds of data only (say -1 for full file, def=-1)", default=None)
    g2.add_argument("-ns", type=int, help="Process only x samples of the data (say -1 for full file, def = -1)", default=None)

    args = a.parse_args()
    return args

def main():
    args = get_parser()
    set_up_logging(args.log_level)
    candname = args.C
    if args.C is None:
        basename = args.f
        candname = "".join(basename.split(".")[:-1]) + ".cand"
    run_search(args.f, args.nt, args.max_dm, args.max_boxcar, args.T, candname, args)



if __name__ == '__main__':
    main(args)
