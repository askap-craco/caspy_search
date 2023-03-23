import numpy as np
ssp = 0.5
def cff(f1, f2, fmin, fmax):

    return (f1**-2 - f2**-2) / (fmin**-2 - fmax**-2)


def get_tstart_and_dm_in_samps_for_a_band(tstart, tend):
    '''
    tstart: float
        The time at which the pulse enters the channel from lower freq edge
    tend: float
        The time at which the pulse leaves the channel from higher freq edge
        
    This convention implies that tstart > tend
    
      1     2     3     4
    __|_____|_____|_____|__
    ...\.................
    ....\................
    .....\...............
    ......\..............
    __|_____|_____|_____|__
      1     2     3     4
    
    
    In the above diagram ~'1' is the tend and ~'1.9' is the tstart
    
      1     2     3     4
    __|_____|_____|_____|____
    ......\.................
    .......\................
    ........\...............
    .........\..............
    __|_____|_____|_____|____
      1     2     3     4
    
    In this example, ~'1.5' is tend and ~'2.2' is tstart
    
    '''
    
    if np.floor(tstart) == np.floor(tend):
        tstart_samp = int(np.floor(tstart))
        dm_samp = 0
    else:
        full_samples = np.floor(tstart) - np.ceil(tend)

        fractional_start_samp = tstart - np.floor(tstart)
        fractional_end_samp = 1- (tend - np.floor(tend))
        
        critical_fraction = np.sqrt(full_samples * (full_samples + 1)) - full_samples
        critical_fraction = 0 
                    
        if fractional_start_samp >= critical_fraction:
            tstart_samp = int(tstart)
        else:
            tstart_samp = int(tstart) -1
                        
        if fractional_end_samp >= critical_fraction:
            tend_samp = int(tend)
        else:
            tend_samp = int(tend) + 1
                        
        dm_samp = tstart_samp - tend_samp
    
    return tstart_samp, dm_samp


def FDMT_initialize(din, f_min, f_max, maxDT):
    '''
    Initialises the FDMT iterations
    '''
    [n_f, n_t] = din.shape
    df = ( f_max - f_min ) / n_f

    max_dm_init = int(np.ceil(cff(f_min, f_min + df, f_min, f_max) * maxDT))

    dout = np.zeros((n_f, max_dm_init+1, n_t))

    dout[:, 0, :] = din
    
    #freqs_l = np.arange(f_min, f_max, df)
    #freqs_u = freqs_l + df
    #all_dms = np.arange(maxDT+1)

    #subchannel_delays = np.empty((n_f, all_dms))


    for i_dm in range(1, max_dm_init+1, 1):
        #subchannel_delays = cff(freqs_l, freqs_u, f_min, f_max) * i_dm
        #start_delays = cff(freqs_l, f_min, f_min, f_max) * i_dm
        #end_delays = cff(freqs_u, f_min, f_min, f_max)

        #samp_starts = int(np.floor(start_delays + ssp))
        #samp_ends = int(np.ceil(end_delays + ssp))
    
        #For the 1st DM
        #dout[0, 1, 1] = din[0, 0] + dout[0, 0, 1]
        #dout[0, 1, 2] = din[0, 1] + dout[0, 0, 2]
        #.
        #.
        
        #For the 2nd DM
        #dout[0, 2, 2] = din[0, 0] + dout[0, 1, 2]
        dout[:, i_dm, i_dm:] = din[:, :-i_dm] + dout[:, i_dm -1, i_dm:]
        
    #print(dout[0, :, :])
     
    #global nsamps_added
    #nsamps_added[:max_dm_init+1] = np.arange(max_dm_init+1) + 1
    return dout

def FDMT_iteration(prev_dout, max_dm, n_f, f_min, f_max, i_iter):
    '''
    Does one FDMT iteration
    i_iter needs to start from 1 for 1st iteration. 0th iteration signifies the initialization step
    n_f is the total no of channels i.e. 256 (stays fixed for every fx call)
    '''
    #print("WORKING ON ITERATION:", i_iter, "\nshape of prev_dout = ", prev_dout.shape)
    din = prev_dout
    [nf_in, n_dm_in, n_t] = prev_dout.shape
    nf_out = nf_in // 2

    df_in = (f_max - f_min) / n_f * 2**(i_iter-1)
    df_out = df_in * 2

    max_dm_iter = int(np.ceil(cff(f_min, f_min + df_out, f_min, f_max) * max_dm))
    n_dm_out = max_dm_iter
    
    dout = np.zeros((nf_out, n_dm_out, n_t ))
    #dout = np.zeros((nf_out, 64, n_t ))

    dout[:, 0, :] = din[:, 0, :].reshape(nf_in//2, 2, n_t).sum(axis=1)

    freqs_l_in = np.arange(f_min, f_max, df_in)
    freqs_u_in = freqs_l_in + df_in
    
    freqs_l_out = freqs_l_in[::2]
    freqs_u_out = freqs_u_in[::2]

    global nsamps_added
    print(f"Iteration number = {i_iter}")
    for i_dm in range(1, max_dm_iter):
        for i_outchan in range(nf_out):
            band_f_min = freqs_l_in[i_outchan]
            band_f_mid = freqs_u_in[i_outchan]
            band_f_max = freqs_u_in[i_outchan+1]
            
            tstart_lower_band = ssp
            delay_lower_band = cff(band_f_min, band_f_mid, band_f_min, band_f_max)*i_dm  #+ve
            tstart_upper_band = tstart_lower_band - delay_lower_band   #can be -ve
            tend_upper_band = tstart_lower_band - i_dm    #can be -ve
            
            tstart_lower_band -= np.floor(tend_upper_band)
            tstart_upper_band -= np.floor(tend_upper_band)
            tend_upper_band -= np.floor(tend_upper_band)
            
            tstart_samp_lower_band, dm_lower_band_samps = get_tstart_and_dm_in_samps_for_a_band(tstart_lower_band, tstart_upper_band)
            tstart_samp_upper_band, dm_upper_band_samps = get_tstart_and_dm_in_samps_for_a_band(tstart_upper_band, tend_upper_band)
            
            tstart_diff = tstart_samp_lower_band - tstart_samp_upper_band
            #print("For iter_no =", i_iter,"i_dm=", i_dm, "i_outchan=", i_outchan, "tstart_lower_band =", tstart_lower_band, "tstart_upper_band = ", tstart_upper_band, "dm_upper=", dm_upper_band_samps, "dm_lower=", dm_lower_band_samps, "delays -- ", start_delay_out, mid_delay_out, end_delay_out)    
            #print("i_dm=", i_dm, "i_outchan=", i_outchan, tstart_lower_band, tstart_upper_band, tstart_diff, n_t)
            dout[i_outchan, i_dm, tstart_samp_lower_band : n_t] = din[i_outchan*2, dm_lower_band_samps, tstart_samp_lower_band : n_t] + din[i_outchan*2 +1, dm_upper_band_samps, tstart_samp_upper_band : n_t - tstart_diff]
            print(f"======= {i_dm}, {dm_lower_band_samps + dm_upper_band_samps} =======")
            
            #nsamps_added[i_dm] += dm_upper_band_samps + dm_lower_band_samps + 2
            '''
            if verbose:
                print(f"For channel {2*i_outchan}, For DM = {i_dm} tstart = {tstart_samp_lower_band}, dm = {dm_lower_band_samps}")
                print(f"For channel {2*i_outchan+1}, For DM = {i_dm} tstart = {tstart_samp_upper_band}, dm = {dm_upper_band_samps}")
            '''
    #print("Returning dout with shape", dout.shape)
    '''
    if verbose:
        print("Dout after iteration = \n",dout)
    '''
    return dout

def FDMT(din, f_min, f_max, max_dm):
    '''
    Performs FDMT on din

    Input:
    din: np.ndarray
        A 2-d numpy array with shape (freq, time)
        The first channel should have the lowest frequency
    
    f_min: float
        Freq of the lower edge of the band (Units don't matter)
    f_max: float
        Freq of the upper edge of the band (Units don't matter)
    max_dm: int
        Maximum dispersion delay to correct for (in samples)
    
    Returns:
    dout: numpy.ndarray
        The FDMT transform of the din with shape (1, maxDT, time)
    '''

    print(f_min*1e-6, f_max*1e-6, max_dm)
    [nf, nt] = din.shape

    n_iter = int(np.log2(nf))
    xx = []
    init_din = FDMT_initialize(din, f_min, f_max, max_dm)
    current_dout = init_din
    xx.append(current_dout)
    for i_iter in range(1, n_iter+1):
        current_dout = FDMT_iteration(current_dout.copy(), max_dm, nf, f_min, f_max, i_iter)
        xx.append(current_dout)
    return current_dout, xx


