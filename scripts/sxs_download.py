import sxs
import numpy as np

catalog = sxs.load('catalog', download=False)
df_events = catalog.table
extrapolation_order = 2

filt = np.full(df_events.shape[0], True)
for (i,chi) in enumerate(df_events['reference_dimensionless_spin1']):
    if ((np.nan in chi) or (chi[0]>1e-5) or (chi[1]>1e-5)): filt[i] = False
for (i,chi) in enumerate(df_events['reference_dimensionless_spin2']):
    if ((np.nan in chi) or (chi[0]>1e-5) or (chi[1]>1e-5)): filt[i] = False
df_events_filt = df_events[filt & df_events.object_types.isin(['BHBH'])]

for (i,event) in enumerate(df_events_filt.index):
    print('Event number {}'.format(i))
    try:
        waveform = sxs.load(event + '/Lev/rhOverM', extrapolation_order=extrapolation_order, download=False)
        metadata = sxs.load(event + '/Lev/metadata.json', download=False)
        print('\n')
    except OSError:
        print('\nEvent {} is removed from the analysis.\n'.format(event))
        continue