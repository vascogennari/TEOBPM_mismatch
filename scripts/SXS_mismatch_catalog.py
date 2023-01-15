# Standard python imports
import numpy as np, matplotlib.pyplot as plt
import os
from math import log10, floor

# pyRing internal imports
from   TEOBPM_mismatch_HMs import FittingFactorTimeDomain
import SXS_mismatch_single_event
import sxs

'''
DOCUMENTATION
The script comoutes the mismatch in time domain bewtween TEOBPM
impemented in pyRing and the SXS catalog, for all the simulations in SXS
with initial spins aligned or anti-aligned with the orbital angular momentum.
Also, one can compare both the TEOBResumS and pyRing
implementations of the TEOBPM model.

The output plots show the percent mismatch as a function of the mass ratio q and
the adimensional final spin af, for the selected modes and the following quantities:
complex waveform, two polarizations, amplitude, phase and instantaneous frequency.
Also, there is the option to show the residuals of the above quantities.

In the results folder, there is also a logbook txt file containing the means and
standard deviations of the mismatches found, among with other info.
The list of the SXS simulations used can be specified at the beginning of the
class 'SXS_catalog', to avoid using the entire catalog all the times.

The pyRing wf is generated on the same time array given by SXS, while TEOBResumS
automatically generates his own time array. For the residuals, the TEOBResumS wf
is linearly interpolated.

WARNING: for the moment, to correctly run the code you need to manually deselect
the time delays for the HMs from pyRing. Specifically, in the TEOBPM class
in pyRing/waveform.py, the DeltaT term should be commented in the computation
of multipole_start_time (line 1321).
'''


def return_param_label(param):
    ''' return a string with the name of 
        the parameter in input '''
    if param == 'h':
        label = r'$r(h_{\ell,m})/M$'
    elif param == 'hp':
        label = r'$r(h_{+,\ell m})/M$'
    elif param == 'hp':
        label = r'$r(h_{\times,\ell m})/M$'
    elif param == 'amp':
        label = r'$rA_{\ell m}/M$'
    elif param == 'phi':
        label = r'$\phi_{\ell m}$'
    elif param == 'omg':
        label = r'$M\omega_{\ell m}$'
    return label


class SXS_catalog:

    def __init__(self, flag_params, modes):

        catalog = sxs.load('catalog', download=True)
        df_events = catalog.table  # Pandas Data Frame object

        select_subset_evs = True
        if select_subset_evs:   # set the indices of the subset
            subset = (0,800)

        # select the BHBH events with aligned or anti-aligned spin
        filt = np.full(df_events.shape[0], True)
        precession = 1e-5
        for (i,chi) in enumerate(df_events['reference_dimensionless_spin1']):
            if ((np.nan in chi) or (np.abs(chi[0])>precession) or (np.abs(chi[1])>precession)): filt[i] = False
        for (i,chi) in enumerate(df_events['reference_dimensionless_spin2']):
            if ((np.nan in chi) or (np.abs(chi[0])>precession) or (np.abs(chi[1])>precession)): filt[i] = False
        df_events_filt = df_events[filt & df_events.object_types.isin(['BHBH'])]

        if select_subset_evs:
            df_events_filt = df_events_filt.index[subset[0]:subset[1]]

        evt_params  = ['name', 'm1', 'm2', 'chi1', 'chi2', 'q', 'M', 'nu']
        wf_params   = ['h', 'hp', 'hc', 'amp', 'phi', 'omg']
        models_list = []
        if flag_params['pyRing']:
            models_list.append('pyRing')
        if flag_params['TEOBResumS']:
            models_list.append('TEOBResumS')

        self.mismatch_catalog_dictionary = {}
        self.mismatch_catalog_dictionary['event_params'] = {x: np.empty(0) for x in evt_params}
        self.mismatch_catalog_dictionary['mismatch'] = {'{}'.format(x): {'{}{}'.format(y[0], y[1]): {z: np.empty(0) for z in wf_params} for y in modes} for x in models_list}
        self.mismatch_catalog_dictionary['event_params']['af'] = np.empty(0)    # initialize final spin item

        self.mode_evs_filt = {'{}'.format(x): {'{}{}'.format(y[0], y[1]): np.empty(0).astype(bool) for y in modes} for x in models_list}
        self.high_mism_evs = {'{}'.format(x): {'{}{}'.format(y[0], y[1]): {} for y in modes} for x in models_list}
        self.mism_stats    = {'{}'.format(x): {'{}{}'.format(y[0], y[1]): {z: {'mean': {}, 'std': {}} for z in wf_params} for y in modes} for x in models_list}
        self.evs_counting  = {'{}'.format(x): {'{}{}'.format(y[0], y[1]): {} for y in modes} for x in models_list}
        self.removed_evs   = ['SXS:BBH:0001', 'SXS:BBH:0002', 'SXS:BBH:1110'] # list of the events removed from the analysis

        for (i,event) in enumerate(df_events_filt):
            if not event in self.removed_evs:
                print('\nEvent {}/{}'.format(i,len(df_events_filt)))
                try:
                    single_event_SXS_mismatch = SXS_mismatch_single_event.SXS_mismatch(event, flag_params, modes)
                except (OSError, IndexError):
                    print('\nEvent {} is removed from the analysis.\n'.format(event))
                    self.removed_evs.append(event)
                    continue

                output_dictionaries = single_event_SXS_mismatch.return_dictionaries()
                event_params = single_event_SXS_mismatch.return_event_params()

                for par in evt_params:
                    self.mismatch_catalog_dictionary['event_params'][par] = np.append(self.mismatch_catalog_dictionary['event_params'][par], event_params[par])
                self.mismatch_catalog_dictionary['event_params']['af'] = np.append(self.mismatch_catalog_dictionary['event_params']['af'], single_event_SXS_mismatch.return_final_spin())

                wf_sxs = output_dictionaries['SXS']['RD']
                for model in models_list:
                    wf = output_dictionaries[model]['RD']

                    for mode in modes:
                        lm = '{}{}'.format(mode[0], mode[1])

                        # build the filter to remove the events with zero HMs
                        if wf[lm]['h'][0]<1e-5:
                            self.mode_evs_filt[model][lm] = np.append(self.mode_evs_filt[model][lm], False)
                            tmp = False
                        else:
                            self.mode_evs_filt[model][lm] = np.append(self.mode_evs_filt[model][lm], True)
                            tmp = True

                        for par in wf_params:
                            FF = FittingFactorTimeDomain(wf_sxs[lm][par], wf[lm][par])
                            M  = 1 - FF
                            self.mismatch_catalog_dictionary['mismatch'][model][lm][par] = np.append(self.mismatch_catalog_dictionary['mismatch'][model][lm][par], M)

                            # list the events with high mismatch
                            if ((par=='h') and (M>0.01)):
                                if tmp:
                                    self.high_mism_evs[model][lm][event] = M*100

        for model in models_list:
            for mode in modes:
                lm = '{}{}'.format(mode[0], mode[1])
                for par in wf_params:
                    try:
                        std_temp = np.std(self.mismatch_catalog_dictionary['mismatch'][model][lm][par][self.mode_evs_filt[model][lm]])*100
                        approx_temp = -int(floor(log10(abs(std_temp))))
                        self.mism_stats[model][lm][par]['mean'] = round(np.mean(self.mismatch_catalog_dictionary['mismatch'][model][lm][par][self.mode_evs_filt[model][lm]])*100, approx_temp)
                        self.mism_stats[model][lm][par]['std']  = round(np.std(self.mismatch_catalog_dictionary['mismatch'][model][lm][par][self.mode_evs_filt[model][lm]])*100, approx_temp)
                    except (ValueError): continue
                self.evs_counting[model][lm] = len(self.mismatch_catalog_dictionary['mismatch'][model][lm]['h'][self.mode_evs_filt[model][lm]])

    def return_catalog_mismatch(self):
        return self.mismatch_catalog_dictionary
    
    def return_mismatch_statistics(self):
        return self.mism_stats, self.evs_counting

    def return_events_logbook(self):
        return self.removed_evs, self.high_mism_evs, self.mode_evs_filt



if __name__=='__main__':
# -------------------------------------------------- #
    # select the modes (if all_modes flag is off)
    modes = [(2,2), (3,3)]

    # FLAGS
    pyRing     = 1      # use pyRing
    TEOBResumS = 0      # use TEOBResumS

    all_modes = 1       # select all available modes for the analysis

    plot      = 1       # 1 plot each model separately, 2 plot both models together. 0 = 1 and 2
    all_param = 1       # plot the mismatch of also amplitude, phase and frequency
    residuals = 0       # FIXME: not fully implemented
# -------------------------------------------------- #

    if all_modes:   # available modes
        modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (4,2)]

    params = ['h']
    if all_param:
        params.extend(['amp', 'phi', 'omg'])

    if (TEOBResumS and not pyRing):  # cannot use af for TEOBResumS
        raise ValueError('To use TEOBResumS you also need to slect pyRing, otherwise the final spin is not defined. Exiting...')

    config_flags = {'SXS':         True,    # <--- do not change
                    'pyRing':      pyRing,
                    'TEOBResumS':  TEOBResumS,
                    'interp':      True}    # <--- do not change

    SXS_catalog_mismatch = SXS_catalog(config_flags, modes)
    mismatch_catalog_dictionary = SXS_catalog_mismatch.return_catalog_mismatch()
    removed_events, high_mismatch_events, mode_event_filter = SXS_catalog_mismatch.return_events_logbook()
    mismatch_statistics, events_counting = SXS_catalog_mismatch.return_mismatch_statistics()

    # set working directories and paths
    work_path  = os.getcwd()
    if not os.path.exists('SXS_mismatch_catalog_results'):
        os.makedirs('SXS_mismatch_catalog_results')
    results_path  = os.path.join(work_path, 'SXS_mismatch_catalog_results')

    # save input parameters in txt file
    with open(os.path.join(results_path, 'logbook.txt'), 'w') as file:
        file.write('Number of events analysed:\n')
        for model in events_counting:
            file.write('{}\n'.format(model))
            for lm in events_counting[model]:
                file.write('\t{}:\t{}\n'.format(lm, events_counting[model][lm]))
        file.write('\nEvents removed from the analysis:\n{}\n\n\n'.format(removed_events))
        file.write('Mismatch statistics [mean \pm standard_deviation] of MM%:\n')
        for model in mismatch_statistics:
            file.write('{}\n'.format(model))
            for lm in mismatch_statistics[model]:
                file.write('{}\n'.format(lm))
                for par in mismatch_statistics[model][lm]:
                    file.write('\t{}:\t{}\t\pm\t{}\n'.format(par, mismatch_statistics[model][lm][par]['mean'], mismatch_statistics[model][lm][par]['std']))
        file.write('\n\nEvents with high percent mismatch in r(h_lm)/M:\n')
        for model in high_mismatch_events:
            file.write('{}\n'.format(model))
            for lm in high_mismatch_events[model]:
                file.write('\t{}\n'.format(lm))
                for event in high_mismatch_events[model][lm]:
                    file.write('\t\t{1:}: {2:.{0}f}\n'.format(0, event.replace('SXS:BBH:',''), high_mismatch_events[model][lm][event]))
                file.write('\n')


    ''' ------------- '''
    ''' PLOTS SECTION '''
    ''' ------------- '''
    X = mismatch_catalog_dictionary['event_params']['q']
    Y = mismatch_catalog_dictionary['event_params']['af']

    for par in params:
        par_label = return_param_label(par)

        if (plot==0 or plot==1):
            for mode in modes:
                lm = '{}{}'.format(mode[0], mode[1])
                for model in mismatch_catalog_dictionary['mismatch']:
                    fig  = plt.figure()
                    ax   = plt.axes(projection ='3d')
                    cmap = plt.get_cmap('RdYlBu_r')

                    X = mismatch_catalog_dictionary['event_params']['q'][mode_event_filter[model][lm]]
                    Y = mismatch_catalog_dictionary['event_params']['af'][mode_event_filter[model][lm]]
                    Z = mismatch_catalog_dictionary['mismatch'][model][lm][par][mode_event_filter[model][lm]]
                    sctt = ax.scatter3D(X, Y, Z*100, c=Z*100, s=10, cmap=cmap, label='{}'.format(model))

                    sm = plt.cm.ScalarMappable(cmap=cmap)
                    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=10, cmap=cmap, pad=0.1)
                    ax.set_title('MODE {mode}  -  SXS-{model} Mismatch {label}'.format(mode=lm, model=model, label=par_label))
                    ax.set_xlabel(r'q')
                    ax.set_ylabel(r'$a_f$')
                    ax.set_zlabel(r'$MM\%$')

                    ax.grid(linestyle='--')
                    fig.tight_layout()
                    plt.savefig(os.path.join(results_path, 'SXS-{}_mismatch_{}_{}.pdf'.format(model, lm, par)))
            
        if (plot==0 or plot==2):
            for mode in modes:
                lm  = '{}{}'.format(mode[0], mode[1])
                fig = plt.figure()
                ax  = plt.axes(projection ='3d')

                for model in mismatch_catalog_dictionary['mismatch']:
                    if model=='pyRing':
                        color  = '#426A0F'
                        marker = 'o'
                    if model=='TEOBResumS':
                        color  = '#1471c0'
                        marker = '*'

                    X = mismatch_catalog_dictionary['event_params']['q'][mode_event_filter[model][lm]]
                    Y = mismatch_catalog_dictionary['event_params']['af'][mode_event_filter[model][lm]]
                    Z = mismatch_catalog_dictionary['mismatch'][model][lm][par][mode_event_filter[model][lm]]
                    sctt = ax.scatter3D(X, Y, Z*100, s=10, c=color, marker=marker, label='{}'.format(model))

                ax.set_title('MODE {mode}  -  SXS Mismatch Comparison {label}'.format(mode=lm, label=par_label))
                ax.set_xlabel(r'q')
                ax.set_ylabel(r'$a_f$')
                ax.set_zlabel(r'$MM\%$')

                ax.legend()
                ax.grid(linestyle='--')
                fig.tight_layout()
                plt.savefig(os.path.join(results_path, 'SXS_comparison_mismatch_{}_{}.pdf'.format(lm, par)))

                if residuals:   # plot residuals between pyRing and TEOBResumS
                    fig  = plt.figure()
                    ax   = plt.axes(projection ='3d')
                    cmap = plt.get_cmap('RdYlBu_r')

                    X = mismatch_catalog_dictionary['event_params']['q'][mode_event_filter[model][lm]]
                    Y = mismatch_catalog_dictionary['event_params']['af'][mode_event_filter[model][lm]]
                    Z_pyr = mismatch_catalog_dictionary['mismatch']['pyRing'][lm]['h'][mode_event_filter['pyRing'][lm]]
                    Z_res = mismatch_catalog_dictionary['mismatch']['TEOBResumS'][lm][par][mode_event_filter['TEOBResumS'][lm]]
                    Z = (Z_res-Z_pyr)/Z_pyr
                    sctt = ax.scatter3D(X, Y, Z*100, c=Z*100, s=10, cmap=cmap, label='M residuals')

                    ax.set_title('MODE {mode}  -  Residuals SXS Mismatch Comparison {label}'.format(mode=lm, label=par_label))
                    ax.set_xlabel(r'q')
                    ax.set_ylabel(r'$a_f$')
                    ax.set_zlabel(r'$MM\%$')

                    ax.legend()
                    ax.grid(linestyle='--')
                    fig.tight_layout()
                    plt.savefig(os.path.join(results_path, 'SXS_comparison_mismatch_{}_{}_residuals.pdf'.format(lm, par)))

#plt.show()
plt.close()