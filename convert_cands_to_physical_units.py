import numpy as np
from Cands_handler import Cands_handler as ch
import pandas as pd
import argparse

def main():
    for candfile in args.cands:
        outname = candfile.strip(".cand") + (".pcand")
        cands = pd.read_csv(args.cands, skiprows=1, skipfooter=1, sep="\s+", header = 0)
        cands = np.asarray(cands)
        final_cands = ch.add_physical_units_columns(cands = cands, fbottom=args.fbottom, df = args.df, nf = args.nchans, tsamp=args.tsamp)
        ch.write_cands(final_cands)
        ch.f.close()

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('cands', type=str, nargs='+', help="Path to the cand files")
    a.add_argument('-nchans', type=int, help="No of freq channels (def = 336)", default=336)
    a.add_argument('-fbottom', type=float, help="Bottom edge of band in MHz (def = 1152.5)", default=1152.5)
    a.add_argument('-df', type=float, help="BW in MHz (def = 1)", default=1)
    a.add_argument('-tsamp', type=float, help="Tsamp in seconds (default=0.00126646875)", default=0.00126646875)
    args=a.parse_args()
    main()