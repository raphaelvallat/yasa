import yasa
import pandas as pd
data = pd.Series([-0.5, -0.7, -0.3, 0.1, 0.15, 0.3, 0.55],
                 index=['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz'])
fig = yasa.topoplot(data, vmin=-1, vmax=1, n_colors=8)
