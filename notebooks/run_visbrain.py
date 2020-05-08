"""This file shows how to use YASA in combination with Visbrain.
"""
import numpy as np
from visbrain.gui import Sleep
from yasa import spindles_detect, sw_detect

# Load the data and hypnogram
data = np.load('data_full_6hrs_100Hz_Cz+Fz+Pz.npz').get('data')
ch_names = ['Cz', 'Fz', 'Pz']
hypno = np.load('data_full_6hrs_100Hz_hypno.npz').get('hypno')

# Initialize a Visbrain.gui.Sleep instance
sl = Sleep(data=data, channels=ch_names, sf=100, hypno=hypno)


# Define spindles function
def fcn_spindle(data, sf, time, hypno):
    """Replace Visbrain built-in spindles detection by YASA algorithm.
    See http://visbrain.org/sleep.html#use-your-own-detections-in-sleep
    """
    # Apply on the full recording
    # sp = spindles_detect(data, sf).summary()
    # NREM sleep only
    sp = spindles_detect(data, sf, hypno=hypno).summary()
    return (sp[['Start', 'End']].values * sf).astype(int)


# Define slow-waves function
def fcn_sw(data, sf, time, hypno):
    """Replace Visbrain built-in slow-wave detection by YASA algorithm.
    """
    # On N2 / N3 sleep only
    # Note that if you want to apply the detection on N3 sleep only, you should
    # use sw_detect(..., include=(3)).summary()
    sw = sw_detect(data, sf, hypno=hypno).summary()
    return (sw[['Start', 'End']].values * sf).astype(int)


# Replace the native Visbrain detections
sl.replace_detections('spindle', fcn_spindle)
sl.replace_detections('sw', fcn_sw)

# Launch the Graphical User Interface
sl.show()
