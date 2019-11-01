import yasa
import numpy as np
# In the next lines, we're loading the data from GitHub.
import requests
from io import BytesIO
r = requests.get('https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_Cz%2BFz%2BPz.npz', stream=True)
npz = np.load(BytesIO(r.raw.read()))
data = npz.get('data')[0, :]
sf = 100
# Load the 30-sec hypnogram and upsample to data
hypno = np.loadtxt('https://raw.githubusercontent.com/raphaelvallat/yasa/master/notebooks/data_full_6hrs_100Hz_hypno_30s.txt')
hypno = yasa.hypno_upsample_to_data(hypno, 1/30, data, sf)
fig = yasa.plot_spectrogram(data, sf, hypno, cmap='Spectral_r')
