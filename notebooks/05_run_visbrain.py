"""This file shows how to use YASA in combination with Visbrain.
"""
import numpy as np
from visbrain.gui import Sleep
from yasa import spindles_detect

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
    # sp = spindles_detect(data, sf)

    # Alternatively if you want to apply the detection only on NREM sleep
    sp = spindles_detect(data, sf, hypno=hypno)

    return (sp[['Start', 'End']].values * sf).astype(int)


# Replace the native Visbrain detection
sl.replace_detections('spindle', fcn_spindle)

# Launch the Graphical User Interface
sl.show()
