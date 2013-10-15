from scipy import linspace, io
from pylab import figure
from cmath import phase
from math import cos
import numpy as np
import pdb
from operator import itemgetter, attrgetter
import datetime

def load_data(file = "H4/Tagged_Training_07_26_1343286001.mat"):
    ''' Load the .mat files. '''
    #testData = io.loadmat('data/H4/Testing_09_13_1347519601.mat', struct_as_record=False, squeeze_me=True)
    
    #file = "H4/Tagged_Training_07_26_1343286001"
    #file = "H4/Tagged_Training_07_27_1343372401"
    #file = "H1/Tagged_Training_04_13_1334300401"

    taggingData = io.loadmat(file, struct_as_record=False, squeeze_me=True)
    #taggingInfoData = io.loadmat('data/H4/AllTaggingInfo.mat', struct_as_record=False, squeeze_me=True)

    # Extract tables
    buf = taggingData['Buffer']
#    pdb.set_trace()
    d = DataStore()
    LF1V = buf.LF1V
    LF1I = buf.LF1I
    LF2V = buf.LF2V
    LF2I = buf.LF2I

    # L1 and L2 time ticks occur every 0.166s.
    d.L1_TimeTicks = buf.TimeTicks1
    d.L2_TimeTicks = buf.TimeTicks2
    d.HF            = buf.HF
                             
    d.HF_TimeTicks = buf.TimeTicksHF
     
    d.taggingInfo = buf.TaggingInfo
     
    # Calculate power by convolution
    L1_P = LF1V * LF1I.conjugate()
    L2_P = LF2V * LF2I.conjugate()
    
    L1_ComplexPower = L1_P.sum(axis=1)
    L2_ComplexPower = L2_P.sum(axis=1)
    
    # Extract components
    d.L1_Real = L1_ComplexPower.real
    d.L1_Imag = L1_ComplexPower.imag
    L1_App  = abs(L1_ComplexPower)
    d.L2_Real = L2_ComplexPower.real
    d.L2_Imag = L2_ComplexPower.imag
    L2_App  = abs(L2_ComplexPower)

    L1_Pf = [cos(phase(L1_P[i,0])) for i in range(len(L1_P[:,0]))]
    L2_Pf = [cos(phase(L2_P[i,0])) for i in range(len(L2_P[:,0]))]
    d.L1_Pf = np.array(L1_Pf,dtype='float64')
    d.L2_Pf = np.array(L2_Pf,dtype='float64')

    d.start = d.L1_TimeTicks[0]
    d.end = d.L1_TimeTicks[-1]

    print("start: ")
    print(date_str(d.start))
    print("end : ")
    print(date_str(d.end))
    return d

class DataStore:
    ''' Container for the EMI data from a single time sample. '''
    def __init__(self):
        pass

def date_str(stamp):
    ''' Converts a UNIX timestamp to a readable UCT date and time. The
    experiment is run at UCT -8, so each sample starts at midnight. '''
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S')

def add_devices(ax, d, timeticks, bottom=300, step=300):
    '''
    Add a green line for every device. '''
    for i in range(len(d.taggingInfo)):
        if d.taggingInfo[i, 2]>= timeticks[0] and d.taggingInfo[i, 3] <=\
        timeticks[-1]:
            ax.plot([d.taggingInfo[i,2],d.taggingInfo[i,3]], [i*step+bottom,i*step+bottom], color=(0,1,0,0.5), linewidth=10)
            str1 = '%s' % d.taggingInfo[i,1]
            ax.text(timeticks[0],step*i+bottom, str1)


HF_tick_size = 1.06
LF_tick_size = 0.16648

lf_tick_60_min = int(60*60/0.16648)
lf_am_12 = int(12*60*60/0.166)
hf_tick_60_min = int(60*60/HF_tick_size)
hf_am_12 = int(12*60*60/HF_tick_size)

def device_sample_all(d, min_bin = 0, max_bin = 4095):
    ''' For each device, plot a sample of the data from the period(s) when it's
    on. '''
    for i in range(40):
        device_sample(d, i, min_bin = min_bin, max_bin = max_bin)

def device_sample(d, device_no, min_bin = 0,
                             max_bin = 4095, buffer_seconds = 60):
    ''' Plot a sample of the data from the period when device number device_no
    is on. '''
    range_subset = []
    for i in range(len(d.taggingInfo)):
        if d.taggingInfo[i,0] == device_no:
            name = d.taggingInfo[i, 1]
            start = d.taggingInfo[i, 2]
            end = d.taggingInfo[i, 3]
            range_set = []
            for (j, t) in enumerate(d.HF_TimeTicks):
                if start - buffer_seconds / HF_tick_size  < t < end +\
                buffer_seconds / HF_tick_size:
                    range_set.append(j)
            if len(range_set) > 0:
                range_subset.append((range_set, start, end))
            print("start: {0} end: {1}".format(start, end))
    if len(range_subset) >0:
        print("No. slices: {0}".format(len(range_subset)))
        fig = figure(10+device_no)
        for (j, r) in enumerate(range_subset):
            (r,s,e) = r
            ax1 = fig.add_subplot("1"+ str(len(range_subset)) +str(j+1))
            ax1.set_title("{0} ({1})".format(name, device_no))
            ax1.set_xlabel("\'Time\'")
            HF_subset = d.HF[min_bin: max_bin,r]
            ax1.imshow(HF_subset, aspect = 0.1)

def smart_plot(d,
               start_time, # Unix timestamp
               period_length,
               L1_real = True, # Flags to show different plots
               L1_imaginary = True,
               L1_factor = True,
               L2_real = False,
               L2_imaginary = False,
               L2_factor = False,
               HF = True,
               min_bin = 50, # For the HF plot
               max_bin = 150,
               show_device_labels = True):
    fig = figure(1)
    fig.clf()
    num_plots = 0
    if L1_real: num_plots += 1
    if L1_imaginary: num_plots += 1
    if L1_factor: num_plots += 1
    if L2_real: num_plots += 1
    if L2_imaginary: num_plots += 1
    if L2_factor: num_plots += 1
    if HF: num_plots += 1
    plot_counter = 1
    # Low frequency time ticks of interest
    # Not designed to be fast.
    subset = []
    for (i, t) in enumerate(d.L1_TimeTicks):
        if start_time <= t <= start_time + period_length:
            subset.append(i)
    subset = np.array(subset)
    
    # High frequency time ticks of interest
    # Not designed to be fast.
    hf_subset = []
    for (i, t) in enumerate(d.HF_TimeTicks):
        if start_time <= t <= start_time + period_length:
            hf_subset.append(i)
    hf_subset = np.array(hf_subset)

    if L1_real:
        ax_L1_real = fig.add_subplot(num_plots, 1, plot_counter)
        ax_L1_real.plot(d.L1_TimeTicks[subset], d.L1_Real[subset], color='blue')
        ax_L1_real.set_title('Real Power Phase 1')
        ax_L1_real.set_ylabel("W")
        ax_L1_real.autoscale(tight = True)
        plot_counter += 1
    if L1_imaginary:
        ax_L1_imaginary = fig.add_subplot(num_plots, 1, plot_counter)
        ax_L1_imaginary.plot(d.L1_TimeTicks[subset], d.L1_Imag[subset], color='blue')
        ax_L1_imaginary.set_title('Imaginary/Reactive Power Phase 1')
        ax_L1_imaginary.set_ylabel("Var")
        ax_L1_imaginary.autoscale(tight = True)
        plot_counter += 1
    if L1_factor:
        ax_L1_factor = fig.add_subplot(num_plots, 1, plot_counter)
        ax_L1_factor.plot(d.L1_TimeTicks[subset],d.L1_Pf[subset])
        ax_L1_factor.set_title('Power Factor Phase 1');
        ax_L1_factor.autoscale(tight = True)
        plot_counter += 1
    if L2_real:
        ax_L2_real = fig.add_subplot(num_plots, 1, plot_counter)
        ax_L2_real.plot(d.L2_TimeTicks[subset], d.L2_Real[subset], color='blue')
        ax_L2_real.set_title('Real Power Phase 2')
        ax_L2_real.autoscale(tight = True)
        plot_counter += 1
    if L2_imaginary:
        ax_L2_imaginary = fig.add_subplot(num_plots, 1, plot_counter)
        ax_L2_imaginary.plot(d.L2_TimeTicks[subset], d.L2_Imag[subset], color='blue')
        ax_L2_imaginary.set_title('Imaginary/Reactive Power Phase 2')
        ax_L2_imaginary.autoscale(tight = True)
        plot_counter += 1
    if L2_factor:
        ax_L2_factor = fig.add_subplot(num_plots, 1, plot_counter)
        ax_L2_factor.plot(d.L2_TimeTicks[subset],d.HFL2_Pf[subset])
        ax_L2_factor.set_title('Power Factor Phase 2');
        ax_L2_factor.autoscale(tight = True)
        plot_counter += 1
    if HF:
        ax_HF = fig.add_subplot(num_plots, 1, plot_counter)
        ax_HF.imshow(d.HF[min_bin: max_bin, hf_subset],
                    aspect = float(len(hf_subset))/(max_bin - min_bin)/10.0)
        ax_HF.set_title('EMI spectogram')
        l = len(hf_subset)
        t = np.arange(0, len(hf_subset), int(len(hf_subset)/10))
        plot_counter += 1
    #fig.tight_layout()
    
    
    ax_L1_factor.set_xlabel('Unix Timestamp');
    if show_device_labels:
        add_devices(ax_L1_real, d, d.L1_TimeTicks[subset])
    return fig 


def total_on_time(d):
    ''' Returns the length of time each device in the training set was on for.
    '''
    times = np.zeros(40)
    names = [None] * 40
    for i in range(len(d.taggingInfo)):
        times[d.taggingInfo[i, 0]] += d.taggingInfo[i, 3] - d.taggingInfo[i, 2]
        names[d.taggingInfo[i, 0]] = d.taggingInfo[i, 1]
    appliances = list(zip(names, times))
    for i in reversed(range(len(appliances))):
        if appliances[i][0] == None:
            del appliances[i]

    appliances.sort(key = itemgetter(1), reverse = True)
    for (n, t) in appliances:
        print("{0}: {1}s".format(n, t))
    return appliances

def hf_time_slice(d, stamp):
    ''' Plots spectogram at the next time after the timestamp. '''
    for (i, t) in enumerate(d.HF_TimeTicks):
        if t > stamp:
            slice = i
    HF_slice = d.HF[:, i]
    assert(len(HF_slice) == 4096)
    fig = figure(972)
    ax1 = fig.add_subplot(111)
    ax1.plot(range(4096), HF_slice)
    ax1.autoscale(tight = True)
    return fig

def local_least_squares(d, 
                        stamp,
                        peak_freqency_bucket,
                        window_size):
    ''' Let's try something simpler before implementing this. '''

def predict_kitchen_dimmer(d):
    ''' Hopefully a prototype for a more general prediction algorithm. ''' 
    # The kitchen dimmer creates a peak around bin 4000.
    # Bin 4000 is the 96th bin from the end.
    # Each bin corresponds to 244 Hz, so the kitchen dimmer has a
    # peak at around 24kHz.
 
    power_in_range = np.zeros(len(d.HF_TimeTicks))
    for (i, t) in enumerate(d.HF_TimeTicks):
        # Take an average over neighbouring bins.
        power_in_range[i] = sum(d.HF[3999:4002, i])/3.0
        
    fig = figure(222)
    ax1 = fig.add_subplot(111)
    ax1.plot(d.HF_TimeTicks, power_in_range)

def bin_data(d):
    ''' Puts the HF data into bins in time and frequency space. The idea is that we might realize a particular device has been turned on if there is suddenly much more power in a particular bin. '''
    # HF has shape 4096 x 81000
    freq_bins = 2048
    time_bins = 1000

    freqs = len(d.HF)
    times = len(d.HF[0])
    f_bin_size = freqs // freq_bins 
    # Don't worry for now that it doesn't fit exactly.
    t_bin_size = times // time_bins 
    
    HF_binned = np.zeros((freq_bins, time_bins))
    for i in range(freq_bins):
        print(i)
        for j in range(time_bins):
            HF_binned[i, j] = sum(sum(d.HF[i * f_bin_size: (i+1) * f_bin_size,\
                                  j * t_bin_size : (j+1) * t_bin_size]))
    np.save("HF_binned" + str(freq_bins) + "-" +str(time_bins), HF_binned) 

def hf_plot(d):
    fig = figure(1)
    ax1 = fig.add_subplot(111)
    hf_10_min = 10 * 60 / HF_tick_size
    ax1.imshow(d.HF[:,  hf_am_12: hf_am_12 +  hf_10_min],
               aspect = 0.07 )
    return fig

def lf_plot(d):
    # subset is the range of indices of d.L1_TimeTicks to plot
    subset = np.array(xrange(lf_am_12,lf_am_12 + lf_tick_60_min))
     
    fig = figure(2)
    fig.set_dpi(150)
    fig.set_size_inches(18.5,50.5)
    # Plot real power consumption
    ax1 = fig.add_subplot(411)
    #ax1.set_xlim(d.d.L1_TimeTicks[0],d.d.L1_TimeTicks[1])
    ax1.plot(d.L1_TimeTicks[subset], d.L1_Real[subset], color='blue')
    ax1.set_title('Real Power (W) and device ON time')
    add_devices(ax1,d,d.L1_TimeTicks[subset])
    # This will draw a green line for every device while it is turned on
     
    fig.subplots_adjust(hspace = 0.4)
       
    # Plot Imaginary/Reactive power (VAR)
    ax2 = fig.add_subplot(412)
    ax2.plot(d.L1_TimeTicks[subset], d.L1_Imag[subset])
    ax2.set_title('Imaginary/Reactive power (VAR)')
    #add_devices(ax2,taggingInfo,d.L1_TimeTicks[subset])
     
    # Plot Power Factor
    ax3 = fig.add_subplot(413)
    ax3.plot(d.L1_TimeTicks[subset],d.L1_Pf[subset])
    ax3.set_title('Power Factor');
    ax3.set_xlabel('Unix Timestamp');
