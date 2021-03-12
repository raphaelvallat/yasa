import yasa
import pandas as pd
data = pd.Series([4, 8, 7, 1, 2, 3, 5],
                 index=['F4', 'F3', 'C4', 'C3', 'P3', 'P4', 'Oz'],
                 name='Values')
fig = yasa.topoplot(data, title='My first topoplot')
