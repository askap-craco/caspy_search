#Cell 1:

#%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
'''
from Visibility_injector.inject_in_fake_data import FakeVisibility

#Cell 2:

from craft.craco_plan import PipelinePlan
from craft import uvfits

f = uvfits.open("/home/gup037/tmp/frb_d0_t0_a1_sninf_lm00.fits")
plan = PipelinePlan(f, "--ndm 64 --nt 1024")

FV = FakeVisibility(plan, injection_params_file="/data/craco/gup037/test_runs_of_craco_pipeline/injections/tmp.yml")


#Cell 3:

data_gen = FV.get_fake_data_block()
iblk = next(data_gen)
iblk.shape

#iblk[:, :, 600] = 1.0 * 0.1


#Cell 4:

plt.figure()
plt.imshow(np.abs(iblk).sum(axis=0), aspect='auto')


#Cell 5:

import numpy as np
ssp = 0.5 
#nsamps_added = np.zeros(1024)

for_dm = 4
verbose = False
'''
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
    if verbose:
        print(dout[0, :, :])
     
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
            if verbose:
                print(f"For channel {2*i_outchan}, For DM = {i_dm} tstart = {tstart_samp_lower_band}, dm = {dm_lower_band_samps}")
                print(f"For channel {2*i_outchan+1}, For DM = {i_dm} tstart = {tstart_samp_upper_band}, dm = {dm_upper_band_samps}")
    #print("Returning dout with shape", dout.shape)
    if verbose:
        print("Dout after iteration = \n",dout)
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



'''

#Cell 6:

y, xx = FDMT(np.abs(iblk).sum(axis=0), plan.fmin - plan.foff/2, plan.fmax + plan.foff/2, 512)


#Cell 7:


plt.figure()
plt.imshow(y.squeeze(), aspect='auto', interpolation='none')

#Cell 8:
'''
import time 
def barack_FDMT(Image, f_min, f_max,maxDT ,dataType, Verbose = True):
    """
    This function implements the  FDMT algorithm.
    Input: Input power matrix I(f,t)
           f_min,f_max are the base-band begin and end frequencies.
                   The frequencies should be entered in MHz 
           maxDT - the maximal delay (in time bins) of the maximal dispersion.
                   Appears in the paper as N_{\Delta}
                   A typical input is maxDT = N_f
           dataType - a valid numpy dtype.
                      reccomended: either int32, or int64.
    Output: The dispersion measure transform of the Input matrix.
            The output dimensions are [Input.shape[1],maxDT]
    
    For details, see algorithm 1 in Zackay & Ofek (2014)
    """
    F,T = Image.shape
    f = int(np.log2(F))
    if (F not in [2**i for i in range(1,30)]) or (T not in [2**i for i in range(1,30)]) :
        raise NotImplementedError("Input dimensions must be a power of 2")

    x = time.time()
    State = barack_FDMT_initialization(Image,f_min,f_max,maxDT,dataType)
    print('initialization ended')
    
    for i_t in range(1,f+1):
        State = barack_FDMT_iteration(State,maxDT,F,f_min,f_max,i_t,dataType, Verbose)
    print('total_time:', time.time() - x)
    [F,dT,T] = State.shape;
    DMT= np.reshape(State,[dT,T]);
    return DMT

def barack_FDMT_initialization(Image,f_min,f_max,maxDT,dataType):
    """
    Input: Image - power matrix I(f,t)
        f_min,f_max - are the base-band begin and end frequencies.
            The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
        maxDT - the maximal delay (in time bins) of the maximal dispersion.
            Appears in the paper as N_{\Delta}
            A typical input is maxDT = N_f
        dataType - To naively use FFT, one must use floating point types.
            Due to casting, use either complex64 or complex128.
    Output: 3d array, with dimensions [N_f,N_d0,Nt]
            where N_d0 is the maximal number of bins the dispersion curve travels at one frequency bin
    
    For details, see algorithm 1 in Zackay & Ofek (2014)
    """
    # Data initialization is done prior to the first FDMT iteration
    # See Equations 17 and 19 in Zackay & Ofek (2014)

    [F,T] = Image.shape

    deltaF = (f_max - f_min)/float(F)
    deltaT = int(np.ceil((maxDT-1) *(1./f_min**2 - 1./(f_min + deltaF)**2) / (1./f_min**2 - 1./f_max**2)))

    Output = np.zeros([F,deltaT+1,T],dataType)
    Output[:,0,:] = Image
    
    for i_dT in range(1,deltaT+1):
        Output[:,i_dT,i_dT:] = Output[:,i_dT-1,i_dT:] + Image[:,:-i_dT]
    return Output


def barack_FDMT_iteration(Input,maxDT,F,f_min,f_max,iteration_num,dataType, Verbose = False):
    """
        Input: 
            Input - 3d array, with dimensions [N_f,N_d0,Nt]
            f_min,f_max - are the base-band begin and end frequencies.
                The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
            maxDT - the maximal delay (in time bins) of the maximal dispersion.
                Appears in the paper as N_{\Delta}
                A typical input is maxDT = N_f
            dataType - To naively use FFT, one must use floating point types.
                Due to casting, use either complex64 or complex128.
            iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes (like in FFT)
        Output: 
            3d array, with dimensions [N_f/2,N_d1,Nt]
        where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
        
        For details, see algorithm 1 in Zackay & Ofek (2014)
    """

    input_dims = Input.shape;
    output_dims = list(input_dims);
    
    deltaF = 2**(iteration_num) * (f_max - f_min)/float(F);
    dF = (f_max - f_min)/float(F)
    # the maximum deltaT needed to calculate at the i'th iteration
    deltaT = int(np.ceil((maxDT-1) *(1./f_min**2 - 1./(f_min + deltaF)**2) / (1./f_min**2 - 1./f_max**2)));
    print("deltaT = ",deltaT)
    print("N_f = ",F/2.**(iteration_num))
    print('input_dims', input_dims)
    
    output_dims[0] = output_dims[0]//2;
    
    
    output_dims[1] = deltaT + 1;
    print('output_dims', output_dims)
    Output = np.zeros(output_dims,dataType);
    
    # No negative D's are calculated => no shift is needed
    # If you want negative dispersions, this will have to change to 1+deltaT,1+deltaTOld
    # Might want to calculate negative dispersions when using coherent dedispersion, to reduce the number of trial dispersions by a factor of 2 (reducing the complexity of the coherent part of the hybrid)
    ShiftOutput = 0
    ShiftInput = 0
    T = output_dims[2]
    F_jumps = output_dims[0]
    
    # For some situations, it is beneficial to play with this correction.
    # When applied to real data, one should carefully analyze and understand the effect of 
    # this correction on the pulse he is looking for (especially if convolving with a specific pulse profile)
    if iteration_num>0:
        correction = dF/2.
    else:
        correction = 0
    for i_F in range(F_jumps):
        
        f_start = (f_max - f_min)/float(F_jumps) * (i_F) + f_min;
        f_end = (f_max - f_min)/float(F_jumps) *(i_F+1) + f_min;
        f_middle = (f_end - f_start)/2. + f_start - correction;
        # it turned out in the end, that putting the correction +dF to f_middle_larger (or -dF/2 to f_middle, and +dF/2 to f_middle larger)
        # is less sensitive than doing nothing when dedispersing a coherently dispersed pulse.
        # The confusing part is that the hitting efficiency is better with the corrections (!?!).
        f_middle_larger = (f_end - f_start)/2 + f_start + correction;
        deltaTLocal = int(np.ceil((maxDT-1) *(1./f_start**2 - 1./(f_end)**2) / (1./f_min**2 - 1./f_max**2)))
        
        for i_dT in range(deltaTLocal+1):
            dT_middle = round(i_dT * (1./f_middle**2 - 1./f_start**2)/(1./f_end**2 - 1./f_start**2))
            dT_middle_index = dT_middle + ShiftInput;
            
            dT_middle_larger = round(i_dT * (1./f_middle_larger**2 - 1./f_start**2)/(1./f_end**2 - 1./f_start**2))            
                     
            
            dT_rest = i_dT - dT_middle_larger;
            dT_rest_index = dT_rest + ShiftInput;
            
            i_T_min = 0;
            
            i_T_max = dT_middle_larger
            Output[i_F,i_dT + ShiftOutput,i_T_min:i_T_max] = Input[2*i_F, dT_middle_index,i_T_min:i_T_max];
            
            
            i_T_min = dT_middle_larger;
            i_T_max = T;
            
            
            
            Output[i_F,i_dT + ShiftOutput,i_T_min:i_T_max] = Input[2*i_F, dT_middle_index,i_T_min:i_T_max] + Input[2*i_F+1, dT_rest_index,i_T_min - dT_middle_larger:i_T_max-dT_middle_larger]
    
    return Output




