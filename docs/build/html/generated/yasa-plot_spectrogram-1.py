import yasa
import numpy as np
# In the next 5 lines, we're loading the data from GitHub.
import requests
from io import BytesIO
r = requests.get('https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_Cz%2BFz%2BPz.npz', stream=True)
npz = np.load(BytesIO(r.raw.read()))
data = npz.get('data')[0, :]
sf = 100
fig = yasa.plot_spectrogram(data, sf)
