import numpy as np
import matplotlib.pyplot as plt
from Boxcarer import Boxcar_and_threshold
from Dedisperser import Dedisperser
import time


def run_test():
    fbottom = 1000      #MHz
    df = 1              #MHz
    nchans = 128
    ftop = fbottom + (df * nchans)  #MHz

    max_dm = 1024
    max_boxcar = 16
    nt = 256

    dd = Dedisperser(fmin = fbottom, df = df, nf = nchans, max_dm=max_dm, nt=nt)
    bt = Boxcar_and_threshold(nt = nt, boxcar_history=np.zeros((max_dm, max_boxcar)), keep_last_boxcar=True)

    dm_boxcar_norm_factors = dd.get_rms_normalising_factor(max_boxcar)

    nblocks = 100
    nadded = 0
    inblocks = [np.random.normal(0, 1, nt * nchans).reshape((nchans, nt)) for ii in range(nblocks)]     #noise
    #inblocks = [np.ones((nchans, nt)) for ii in range(nblocks)]     #ones
    
    
    #inblocks = [np.array([np.arange(nt)] * nchans) + ii*nt for ii in range(nblocks)]        #ramp
    #inblocks = [np.array([np.arange(nt)] * nchans) for ii in range(nblocks)]        #saw-tooth

    #variances = np.zeros((max_dm, max_boxcar))
    stds = np.zeros((nblocks, max_dm, max_boxcar))
    boxouts = []
    #mean_boxout = np.zeros((max_dm, nt, max_boxcar))
    nadded = 0
    first_included_block =0 
    for iblock, block in enumerate(inblocks):
        start = time.time()
        dmt_out = dd.get_full_DMT(block)
        dmttime = time.time()
        box_out = bt.run_pure_boxcar(dmt_out)       #Box_out has shape (max_dm, nt, max_boxcar)
        bxtime = time.time()

        print(f"dmt took {dmttime -start}s, boxcar took {bxtime - dmttime}")
        
        #assert np.all(dmt_out[:, :nt] == box_out[:, :, 0]), "The input is not the same as output for boxcar 0"
        if iblock < max_dm //nt + 1:
            continue
            #pass
        if nadded == 0:
            first_included_block = iblock
        #variances += box_out.var(axis=1)   
        #nadded += 1
        stds[iblock] = box_out.std(axis=1)
        #mean_boxout += box_out.copy()
        boxouts.append(box_out)
        nadded += 1
        '''
        plt.figure()
        plt.imshow(block, aspect='auto', interpolation='None')
        plt.title(f"Raw block {iblock}")

        plt.figure()
        plt.imshow(dmt_out, aspect='auto', interpolation='None')
        plt.title(f"DMT out {iblock}")

        plt.figure()
        plt.imshow(box_out[0], aspect='auto', interpolation='None')
        plt.title(f"box out {iblock}")
        plt.figure()
        plt.plot(box_out[0][0], '.')
        #plt.plot(np.arange(1, max_boxcar+1, 1)-1, nchans * np.arange(1, max_boxcar+1, 1))
        '''
        print(iblock)
    
    #plt.show()
    #rmses = np.sqrt(variances / nadded)
    mean_stds = np.mean(stds[first_included_block:nadded+first_included_block], axis=0)
    #mean_boxout /= nadded
    all_boxout_combined = np.concatenate(boxouts, axis=1)

    plt.figure()
    for ibox in range(max_boxcar):
        plt.plot(mean_stds[:, ibox], label=f"boxc={ibox}")
        #plt.plot(mean_boxout.std(axis=1))

    plt.gca().set_prop_cycle(None)
    for ibox in range(max_boxcar):
        plt.plot(all_boxout_combined.std(axis=1)[:, ibox], '.-')
    
    plt.gca().set_prop_cycle(None)
    for ibox in range(max_boxcar):
        plt.plot(dm_boxcar_norm_factors[:, ibox], ls='--')
    
    plt.legend()
    plt.ylabel("Expected rms (Clancy's formula) or Actual rms (measured)")
    plt.xlabel("DM trial")
    plt.title(f"Averaged over {nblocks} blocks")


    plt.figure()
    plt.imshow(mean_stds, aspect='auto', interpolation="None")
    plt.ylabel("DM")
    plt.xlabel("Boxcar")

    plt.figure()
    
    for ibox in range(max_boxcar):
        plt.plot(dm_boxcar_norm_factors[:, ibox] / mean_stds[:, ibox], label=f"boxc={ibox}")

    plt.title(f"std taken for each block, and then averaged over {nblocks} blocks")
    plt.legend()
    plt.ylabel("Ratio of expected rms / actual rms")
    plt.xlabel("DM trial")
    

    plt.figure()

    for ibox in range(max_boxcar):
        plt.plot(dm_boxcar_norm_factors[:, ibox] / all_boxout_combined.std(axis=1)[:, ibox], label=f"boxc={ibox}")


    plt.legend()
    plt.ylabel("Ratio of expected rms / actual rms")
    plt.xlabel("DM trial")
    plt.title(f"Std taken over {nblocks} blocks concatenated together")
    
    plt.show()

if __name__=='__main__':
    run_test()

