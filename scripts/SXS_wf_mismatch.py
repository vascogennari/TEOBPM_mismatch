# Standard python imports
import numpy as np, matplotlib.pyplot as plt
import os

# WFs import
import sxs                      # SXS
import EOBRun_module            # TEOBResumS
import pyRing.waveform as wf    # pyRing
from pyRing.utils import *

# LVC-specific imports
from lalinference.imrtgr.nrutils import *
import lal

'''
DOCUMENTATION
The script compares the match of TEOBPM w.r.t the NR wfs from SXS catalog.
Namely, there can be compared both the TEOBResumS and pyRing
implementations of the TEOBPM model.

The output plots show the comparison of the two polarizations, the amplitude,
phase and instantaneous frequency for the selected modes. There is also the
option to show the residuals of the above quantities.

The pyRing wf is evaluated on the same time array given by SXS, while TEOBResumS
automatically generates his own time array. For the residuals, the TEOBResumS wf
is linearly interpolated.

WARNING: for the moment, to correctly run the code you need to manually deselect
the delayed starting times from pyRing. Specifically, in the TEOBPM class
in pyRing/waveform.py, the DeltaT term should be commented in the computation
of multipole_start_time (line 1321).

#IMPROVEME: implement the mismatch in time-domain and the option to run on the
entire SXS catalog
'''

# --------------------------------------------------- #
# --------------------------------------------------- #
# select init parameters

# name of the SXS simulation:
# https://data.black-holes.org/waveforms/catalog.html
event  = '0376'

# if not all_modes flag, select the modes
modes = [(2,2), (2,1), (3,3)]

# FLAGS
SXS        = 1  # use SXS <---- do not change, for the moment
TEOBResumS = 1  # use TEOBResumS
pyRing     = 1  # use pyRing

IMR_plots  = 0  # plots of full IMR waveform (only SXS, TEOBResumS). if IMR_plots=2, plot only RD
match_plot = 2  # match plots of the selected models for RD. if match_plot=2, also plot the residuals

all_modes  = 0  # select all available modes for the analysis
interp     = 1  # interpolate TEOBResumS wrt SXS time

amp_omg    = 1  # show amplitude and frequency for IMR_plots

# if not SXS flag, select init params <---- not implemented, for the moment
# m1   = 60
# m2   = 20
# chi1 = 0.1
# chi2 = 0.1
# --------------------------------------------------- #
# --------------------------------------------------- #


''' -------------------------- '''
''' AUXILIARY SETTINGS SECTION '''
''' ---------------------------------------------- '''
''' this section contain global auxiliary settings '''
''' for the analysis: please DO NOT CHANGE         '''
''' ---------------------------------------------- '''
'''If SXS is selcted, masses and spins are taken from
   the selected SXS simulation. This values will be also
   used for TEOBResumS and pyRing'''

if SXS:
    # download wf
    extrapolation_order = 2     # 2 is suggested for ringdown
    w = sxs.load('SXS:BBH:' + event + '/Lev/rhOverM', extrapolation_order=extrapolation_order)
    metadata = sxs.load('SXS:BBH:' + event + '/Lev/metadata.json')

    # check for spin consistency: chi_1,2 need to be aligned or anti-aligned
    if ((metadata.reference_dimensionless_spin1[0]>1e-5) or (metadata.reference_dimensionless_spin1[1]>1e-5) or (metadata.reference_dimensionless_spin2[0]>1e-5) or (metadata.reference_dimensionless_spin2[1]>1e-5)):
        raise ValueError('\nThe selected SXS simulation has non-aligned spins. Exiting...')

    # init params
    q    = metadata.reference_mass_ratio
    m1   = metadata.reference_mass1
    m2   = metadata.reference_mass2
    chi1 = metadata.reference_dimensionless_spin1[2]
    chi2 = metadata.reference_dimensionless_spin2[2]

if interp:  # if TEOBResumS is not selected, cannot interpolate TEOBResumS
    if not TEOBResumS:
        interp = 0

M  = m1+m2
q  = m1/m2
nu = q/((1+q)**2)

dist = 450  # distance [Mpc]
iota = 0.0  # inclination [rad]

# available modes
if all_modes:
    modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (4,2)]

# return an array of len(w.LM), with False if the
# mode has not been selected, True otherwise
def modes_filt(modes, NR_modes):
    modes= np.array(modes)
    filt = np.full(len(NR_modes), False)
    i = 0
    for NR_mode in NR_modes:
        for mode in modes:
            if (str(NR_mode) == str(mode)):
                filt[i] = True
        i += 1
    return filt

# return a list of modes using the labeling
# convention of TEOBResumS
def modes_to_k(modes):
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

# return a list of arrays containing the
# important quantities of one single mode
def mode_output(hp, hc, t, sgn):
    amp = np.sqrt(hp**2 + hc**2)
    i   = np.argmax(amp)
    phi = np.unwrap(np.angle(hp - 1j*hc))
    if sgn: 
        phi = phi - phi[i]  # used for TEOBResumS
    else:
        phi = -(phi - phi[i])   # reverse phase sgn to make phase grow
    h   = amp * np.exp(1j*phi)
    hp  = np.real(h)
    hc  = -np.imag(h)   # standard convention h = h+ - ihx
    omg = np.zeros_like(phi)
    omg[1:] = np.diff(phi)/np.diff(t)
    return [hp, hc, amp, phi, omg]

# cut the array to select ringdown
def rd_cut(h, idx):
    return h[idx:]

# set working directories and paths
work_path  = os.getcwd()
if not os.path.exists('SXS_wf_mismatch_plots'):
    os.makedirs('SXS_wf_mismatch_plots')
plot_path  = os.path.join(work_path, 'SXS_wf_mismatch_plots')
dir_name = event + '__q({2:.{0}f})_a({3:.{1}f},{4:.{1}f})'.format(1, 2, q, chi1, chi2)
if not os.path.exists(os.path.join(plot_path, dir_name)):
    os.makedirs(os.path.join(plot_path, dir_name))
event_path = os.path.join(plot_path, dir_name)

# save input parameters in txt file
input_pars = {'name': 'SXS:BBH:'+event,
              'extr_order': extrapolation_order,
              'm1': m1,
              'm2': m2,
              'chi1': chi1,
              'chi2': chi2,
              'q ': q,
              'M': M,
              'nu': nu}
file = open(os.path.join(event_path, '0_input_parameters.txt'), 'w')
file.write('The SXS simulation has the following parameters:\n\n')
for par in input_pars:
    file.write('{} = {}\n'.format(par, input_pars[par]))
file.write('\n\nMasses and spins are evaluated in the reference frame.\n')
file.write('Resolution should be the maximum aivailable, check in metadata if intersted.')
file.close()

dict_names_imr = ['hp', 'hc', 'amp', 'phi', 'omg']
dict_names     = ['hp', 'hc', 'amp', 'phi', 'omg', 't']
figsize_imr    = (10,5)
figsize_match  = (12,7)
fontsize       = 7
fontsize_title = 15
rd_lenght      = 100    # lenght of rd for interpolation

conv_mass_time = lal.MSUN_SI * lal.G_SI / (lal.C_SI**3)                     # mass-time units conversion
conv_mass_dist = lal.MSUN_SI * lal.G_SI / ( lal.PC_SI*1e6 * lal.C_SI**2)    # mass-distance units conversion



''' ----------------------- '''
''' WF CONSTRUCTION SECTION '''
''' ------------------------------------------------------------------ '''
''' build dictionary with wf quantities for SXS, TEOBResmuS and pyRing '''
''' - wf_imr_'model-name' contain the full IMR waveform                '''
''' - wf_'model-name' contain only the ringdown part                   '''
''' ------------------------------------------------------------------ '''
# SXS
# ---------------------------------------------------
if SXS:
    # cut the NR noise at the edges
    idx_ini = w.index_closest_to(metadata.reference_time)
    idx_end = w.index_closest_to(w.max_norm_time() + 100.0)
    idx_rd  = w.index_closest_to(w.max_norm_time())
    w = w[idx_ini:idx_end]

    # select hp and hc
    hp_sxs_lm = np.transpose(w.data.real)
    hc_sxs_lm = -np.transpose(w.data.imag)  # minus sign for convention h+-ihx
    mode_filt = modes_filt(modes, w.LM)
    LM_filt   = w.LM[mode_filt]
    hp_sxs_lm = hp_sxs_lm[mode_filt]  # filter the non selected modes
    hc_sxs_lm = hc_sxs_lm[mode_filt]
    t_sxs = w.t

    # init two dictionaries with all the wf informations for IMR/RD
    wf_imr_sxs = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names_imr} for x in modes}   # IMR
    wf_sxs     = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names} for x in modes}    # RD
    wf_imr_sxs['fm'] = {y: np.empty(0) for y in dict_names_imr}
    wf_imr_sxs['t']  = np.empty(0)

    hp_sxs_fm = np.zeros(len(t_sxs))
    hc_sxs_fm = np.zeros(len(t_sxs))

    for mode in range(len(LM_filt)):
        lm = '{}{}'.format(LM_filt[mode][0], LM_filt[mode][1])
        # IMR
        mode_out = mode_output(hp_sxs_lm[mode], hc_sxs_lm[mode], t_sxs, 0)
        for i in range(len(dict_names_imr)):
            wf_imr_sxs[lm][dict_names_imr[i]] = mode_out[i]
        # RD
        i_sxs = np.argmax(np.sqrt(hp_sxs_lm[mode]**2 + hc_sxs_lm[mode]**2))  # merger time index as peak of the mode
        t_sxs = t_sxs - t_sxs[i_sxs]
        mode_out.append(t_sxs)
        for x in range(len(mode_out)):
            mode_out[x] = rd_cut(mode_out[x], i_sxs)
        for i in range(len(dict_names)):
            wf_sxs[lm][dict_names[i]] = mode_out[i]

        hp_sxs_fm += hp_sxs_lm[mode]
        hc_sxs_fm += hc_sxs_lm[mode]

    mode_out_fm = mode_output(hp_sxs_fm, hc_sxs_fm, t_sxs, 0)
    for i in range(len(dict_names_imr)):
        wf_imr_sxs['fm'][dict_names_imr[i]] = mode_out_fm[i]
    wf_imr_sxs['t'] = t_sxs

# TEOBResumS
# ---------------------------------------------------
if TEOBResumS:
    k = modes_to_k(modes)
    '''if SXS is selected, input values are
       taken from the selected simulation'''
    pars = {
        'M'                  : M,
        'q'                  : q,
        'Lambda1'            : 0., # <---- do not change
        'Lambda2'            : 0., # <---- do not change
        'chi1'               : chi1,
        'chi2'               : chi2,
        'domain'             : 0,      # 0 = time domain
        'arg_out'            : 1,      # Output hlm/hflm. Default = 0
        'use_mode_lm'        : k,      # List of modes to use/output through EOBRunPy
        'srate_interp'       : 4096.,  # srate at which to interpolate. Default = 4096.
        'use_geometric_units': 'yes',  # Output quantities in geometric units. Default = "yes"
        'initial_frequency'  : 35.,    # in Hz if use_geometric_units = "no", else in geometric units
        'interp_uniform_grid': "no",   # Interpolate mode by mode on a uniform grid. Default = "no"
        'distance'           : dist,
        'inclination'        : iota,
        'output_hpc'         : "no",
    }
    '''hlm contain amp and phase of the selected modes in geometric units.
       hp, hp are the polarizations projected on the spherical harmonics
       WARNING: also if 'use_geometric_units' is selected, hp and hc are
                still projected on SH'''
    t_res, hp, hc, hlm_res, dyn = EOBRun_module.EOBRunPy(pars)

    for mode in hlm_res:
        hlm_res['{}'.format(mode)][0] * nu  # TEOBResumS wf in geom units is (r h)/(M nu). need to multiply for nu

    # init two dictionaries with all the wf informations for IMR/RD
    wf_imr_res = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names_imr} for x in modes}   # IMR
    wf_res     = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names} for x in modes}    # RD
    wf_imr_res['fm'] = {y: np.empty(0) for y in dict_names_imr}
    wf_imr_res['t']  = np.empty(0)

    hp_res_fm = np.zeros(len(t_res))
    hc_res_fm = np.zeros(len(t_res))
    j=0

    for mode in k:
        lm = '{}{}'.format(modes[j][0], modes[j][1])

        amp_res_lm = hlm_res['{}'.format(mode)][0] * nu # TEOBResumS wf in geom units is (r h)/(M nu). need to multiply for nu
        phi_res_lm = hlm_res['{}'.format(mode)][1]
        h_res_lm   = amp_res_lm * np.exp(1j*phi_res_lm)
        hp_res_lm  = np.real(h_res_lm)
        hc_res_lm  = -np.imag(h_res_lm)

        # IMR
        mode_out = mode_output(hp_res_lm, hc_res_lm, t_res, 1)
        for i in range(len(dict_names_imr)):
            wf_imr_res[lm][dict_names_imr[i]] = mode_out[i]
        # RD
        i_res = np.argmax(np.sqrt(mode_out[0]**2 + mode_out[1]**2))  # RD start-time index as peak of the mode
        t_res_rd = t_res-t_res[i_res]
        mode_out.append(t_res_rd)
        for x in range(len(mode_out)):
            mode_out[x] = rd_cut(mode_out[x], i_res)
        for i in range(len(dict_names)):
            wf_res[lm][dict_names[i]] = mode_out[i]

        hp_res_fm += hp_res_lm
        hc_res_fm += hc_res_lm
        j+=1

    mode_out_fm = mode_output(hp_res_fm, hc_res_fm, t_res, 1)
    for i in range(len(dict_names_imr)):
        wf_imr_res['fm'][dict_names_imr[i]] = mode_out_fm[i]
    wf_imr_res['t'] = t_res


''' --------------------------- '''
''' FULL WAVEFORM PLOTS SECTION '''
''' -------------------------------------------- '''
''' plots of IMR waveform for SXS and TEOBResmuS '''
''' if IMR_plots=2 plot only the ringdown part   '''
''' -------------------------------------------- '''
if IMR_plots:
    if IMR_plots!=2: # IMR
        for mode in modes:
            if x != 't':
                x = '{}{}'.format(mode[0], mode[1])
                if SXS:
                    plt.figure(figsize=figsize_imr)
                    plt.plot(wf_imr_sxs['t'], wf_imr_sxs[x]['hp'], label=r'$h_{+}$ SXS')
                    plt.plot(wf_imr_sxs['t'], wf_imr_sxs[x]['hc'], label=r'$h_{\times}$ SXS')
                    if amp_omg:
                        amp_max = np.max(wf_sxs[x]['amp'])
                        omg_max = np.max(wf_sxs[x]['omg'])
                        plt.plot(wf_imr_sxs['t'], wf_imr_sxs[x]['amp'], label=r'$A$ SXS')
                        plt.plot(wf_imr_sxs['t'], wf_imr_sxs[x]['omg']*(amp_max/omg_max), label=r'amp-rescaled $\omega$ SXS')
                    plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
                    plt.xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                    plt.ylabel(r'$r(h_{+,\ell m})/M$')
                    plt.grid(linestyle='--')
                    plt.legend(fontsize=fontsize)
                    plt.tight_layout()
                    plt.savefig(os.path.join(event_path, 'IMR_SXS_mode-{}.pdf'.format(x)))
                if TEOBResumS:
                    plt.figure(figsize=figsize_imr)
                    plt.plot(wf_imr_res['t'], wf_imr_res[x]['hp'], label=r'$h_{+}$ TEOBResumS')
                    plt.plot(wf_imr_res['t'], wf_imr_res[x]['hc'], label=r'$h_{\times}$ TEOBResumS')
                    if amp_omg:
                        amp_max = np.max(wf_res[x]['amp'])
                        omg_max = np.max(wf_res[x]['omg'])
                        plt.plot(wf_imr_res['t'], wf_imr_res[x]['amp'], label=r'$A$ TEOBResumS')
                        plt.plot(wf_imr_res['t'], wf_imr_res[x]['omg']*(amp_max/omg_max), label=r'amp-rescaled $\omega$ TEOBResumS')
                    plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
                    plt.xlabel(r'$t$ $[M]$')
                    plt.ylabel(r'$r(h_{+,\ell m})/M$')
                    plt.grid(linestyle='--')
                    plt.legend(fontsize=fontsize)
                    plt.tight_layout()
                    plt.savefig(os.path.join(event_path, 'IMR_TEOBResumS_mode-{}.pdf'.format(x)))

    else:
        for mode in modes:
            x = '{}{}'.format(mode[0], mode[1])
            if SXS:
                # plot of SXS wf directely from catalog
                # plt.figure(figsize=figsize_wf)
                # plt.plot(w.t, w.data.view(float))
                # plt.title("SXS wf from catalog")
                # plt.xlabel('time [M]')
                # plt.grid(linestyle='--')

                plt.figure(figsize=figsize_imr)
                plt.plot(wf_sxs[x]['t'], wf_sxs[x]['hp'], label=r'$h_{+}$ SXS')
                plt.plot(wf_sxs[x]['t'], wf_sxs[x]['hc'], label=r'$h_{\times}$ SXS')
                if amp_omg:
                    amp_max = np.max(wf_sxs[x]['amp'])
                    omg_max = np.max(wf_sxs[x]['omg'])
                    plt.plot(wf_sxs[x]['t'], wf_sxs[x]['amp'], label=r'$A$ SXS')
                    plt.plot(wf_sxs[x]['t'], wf_sxs[x]['omg']*(amp_max/omg_max), label=r'amp-rescaled $\omega$ SXS')
                plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
                plt.xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                plt.ylabel(r'$r(h_{+,\ell m})/M$')
                plt.grid(linestyle='--')
                plt.legend(fontsize=fontsize)
                plt.tight_layout()
                plt.savefig(os.path.join(os.getcwd(), 'plots', event, 'RD_SXS_mode-{}.pdf'.format(x)))
            if TEOBResumS:
                plt.figure(figsize=figsize_imr)
                plt.plot(wf_res[x]['t'], wf_res[x]['hp'], label=r'$h_{+}$ TEOBResumS')
                plt.plot(wf_res[x]['t'], wf_res[x]['hc'], label=r'$h_{\times}$ TEOBResumS')
                if amp_omg:
                    amp_max = np.max(wf_res[x]['amp'])
                    omg_max = np.max(wf_res[x]['omg'])
                    plt.plot(wf_res[x]['t'], wf_res[x]['amp'], label=r'$A$ TEOBResumS')
                    plt.plot(wf_res[x]['t'], wf_res[x]['omg']*(amp_max/omg_max), label=r'amp-rescaled $\omega$ TEOBResumS')
                plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
                plt.xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                plt.ylabel(r'$r(h_{+,\ell m})/M$')
                plt.grid(linestyle='--')
                plt.legend(fontsize=fontsize)
                plt.tight_layout()
                plt.savefig(os.path.join(event_path, 'RD_TEOBResumS_mode-{}.pdf'.format(x)))


''' ------------------- '''
''' MATCH PLOTS SECTION '''
''' ------------------------------------------ '''
''' match plots for SXS, TEOBResmuS and pyRing '''
''' if match_plot=1 also plot the residuals:   '''
''' residuals are computed wrt SXS waveform    '''
''' ------------------------------------------ '''
if match_plot:
    if SXS:
        # init dictionary with all the wf informations
        wf_pyr  = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names} for x in modes}    # RD only
        for mode in modes:
            x = '{}{}'.format(mode[0], mode[1])
            t = wf_sxs[x]['t']

            # pyRing
            # ---------------------------------------------------
            '''for proper working, need to comment DeltaT in
            the starting time of the modes (waveform.pyx)'''
            if pyRing:
                # set initial phase
                phi_0 = wf_sxs[x]['phi'][0]
                '''if SXS is selected, input values are
                taken from the selected simulation'''
                TEOBPM_model = wf.TEOBPM(0.0, # <---- start time, do not change
                                        m1,
                                        m2,
                                        chi1,
                                        chi2,
                                        {mode : phi_0},
                                        dist,
                                        iota,
                                        0.0,
                                        [mode],
                                        0, # <---- full modes flag, do not change
                                        {},
                                        1) # <---- geometric units, do not change
            
                t_pyr = t * (M*conv_mass_time)  # convert time in [s]

                _, _, _, hp_pyr, hc_pyr = TEOBPM_model.waveform(t_pyr) * nu # pyRing wf in geom units is (r h)/(M nu). need to multiply for nu

                mode_out = mode_output(hp_pyr, hc_pyr, t, 0)
                mode_out.append(t)
                for i in range(len(dict_names)):
                    wf_pyr[x][dict_names[i]] = mode_out[i]
                
                wf_pyr[x]['omg'][0] = wf_pyr[x]['omg'][1]   #FIXME: this is a temporary hard-fix to avoid the first point to be zero

            # interpolate TEOBResumS waveform wrt SXS time
            if interp:
                for i in range(len(dict_names)):
                    wf_res[x][dict_names[i]] = np.interp(t, wf_res[x]['t'], wf_res[x][dict_names[i]])

            # plots of (h+, hx, h+-hx, amp, phi, omg) for the selected modes
            fig, axs = plt.subplots(3, 2, figsize=figsize_match)
            if SXS:
                axs[0][0].plot(t, wf_sxs[x]['hp'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                axs[1][0].plot(t, wf_sxs[x]['hc'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                axs[2][0].plot(t, wf_sxs[x]['hp']-wf_sxs[x]['hc'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                axs[0][1].plot(t, wf_sxs[x]['amp'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                axs[1][1].plot(t, wf_sxs[x]['phi'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                axs[2][1].plot(t, wf_sxs[x]['omg'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
            if TEOBResumS:
                if interp:
                    axs[0][0].plot(t, wf_res[x]['hp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[1][0].plot(t, wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[2][0].plot(t, wf_res[x]['hp']-wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[0][1].plot(t, wf_res[x]['amp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[1][1].plot(t, wf_res[x]['phi'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[2][1].plot(t, wf_res[x]['omg'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                else:
                    axs[0][0].plot(wf_res[x]['t'], wf_res[x]['hp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[1][0].plot(wf_res[x]['t'], wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[2][0].plot(wf_res[x]['t'], wf_res[x]['hp']-wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[0][1].plot(wf_res[x]['t'], wf_res[x]['amp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[1][1].plot(wf_res[x]['t'], wf_res[x]['phi'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[2][1].plot(wf_res[x]['t'], wf_res[x]['omg'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
            if pyRing:
                axs[0][0].plot(t, wf_pyr[x]['hp'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                axs[1][0].plot(t, wf_pyr[x]['hc'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                axs[2][0].plot(t, wf_pyr[x]['hp']-wf_pyr[x]['hc'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                axs[0][1].plot(t, wf_pyr[x]['amp'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                axs[1][1].plot(t, wf_pyr[x]['phi'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                axs[2][1].plot(t, wf_pyr[x]['omg'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)

            fig.suptitle(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
            axs[0][0].set_ylabel(r'$r(h_{+,\ell m})/M$')
            axs[1][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
            axs[2][0].set_ylabel(r'$r(h_{+,\ell m}-h_{\times,\ell m})/M$')
            axs[0][1].set_ylabel(r'$rA_{\ell m}/M$')
            axs[1][1].set_ylabel(r'$\phi_{\ell m}$')
            axs[2][1].set_ylabel(r'$M\omega_{\ell m}$')
            axs[2][0].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
            axs[2][1].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
            axs[0][0].set_xlim(-5, 100)
            axs[1][0].set_xlim(-5, 100)
            axs[2][0].set_xlim(-5, 100)
            axs[0][1].set_xlim(-5, 100)
            axs[1][1].set_xlim(-5, 100)
            axs[2][1].set_xlim(-5, 100)
            axs[0][0].set_xticklabels([])
            axs[0][1].set_xticklabels([])
            axs[1][0].set_xticklabels([])
            axs[1][1].set_xticklabels([])
            axs[0][0].grid(linestyle='--')
            axs[1][0].grid(linestyle='--')
            axs[2][0].grid(linestyle='--')
            axs[0][1].grid(linestyle='--')
            axs[1][1].grid(linestyle='--')
            axs[2][1].grid(linestyle='--')
            axs[0][0].legend(fontsize=fontsize)
            axs[1][0].legend(fontsize=fontsize)
            axs[2][0].legend(fontsize=fontsize)
            axs[0][1].legend(fontsize=fontsize)
            axs[1][1].legend(fontsize=fontsize)
            axs[2][1].legend(fontsize=fontsize)
            fig.tight_layout()
            plt.savefig(os.path.join(event_path, 'match_plot_mode-{}.pdf'.format(x)))

            # plots of the difference wrt SXS waveform of
            # (h+, hx, h+-hx, amp, phi, omg) for the selected modes
            if match_plot==2:
                if TEOBResumS:
                    if not interp:
                        raise ValueError('\nCannot plot waveform differences without interpolating TEOBResumS')

                fig, axs = plt.subplots(3, 2, figsize=figsize_match)
                if TEOBResumS:
                    axs[0][0].plot(t, wf_res[x]['hp']-wf_sxs[x]['hp'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[1][0].plot(t, wf_res[x]['hc']-wf_sxs[x]['hc'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[2][0].plot(t, (wf_res[x]['hp']-wf_sxs[x]['hp'])-(wf_res[x]['hc']-wf_sxs[x]['hc']), label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[0][1].plot(t, wf_res[x]['amp']-wf_sxs[x]['amp'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[1][1].plot(t, wf_res[x]['phi']-wf_sxs[x]['phi'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    axs[2][1].plot(t, wf_res[x]['omg']-wf_sxs[x]['omg'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                if pyRing:
                    axs[0][0].plot(t, wf_pyr[x]['hp']-wf_sxs[x]['hp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                    axs[1][0].plot(t, wf_pyr[x]['hc']-wf_sxs[x]['hc'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                    axs[2][0].plot(t, (wf_pyr[x]['hp']-wf_sxs[x]['hp'])-(wf_pyr[x]['hc']-wf_sxs[x]['hc']), label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                    axs[0][1].plot(t, wf_pyr[x]['amp']-wf_sxs[x]['amp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                    axs[1][1].plot(t, wf_pyr[x]['phi']-wf_sxs[x]['phi'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                    axs[2][1].plot(t, wf_pyr[x]['omg']-wf_sxs[x]['omg'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)

                fig.suptitle(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
                axs[0][0].set_ylabel(r'$r(h_{+,\ell m})/M$')
                axs[1][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                axs[2][0].set_ylabel(r'$r(h_{+,\ell m}-h_{\times,\ell m})/M$')
                axs[0][1].set_ylabel(r'$rA_{\ell m}/M$')
                axs[1][1].set_ylabel(r'$\phi_{\ell m}$')
                axs[2][1].set_ylabel(r'$M\omega_{\ell m}$')
                axs[2][0].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                axs[2][1].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                axs[0][0].set_xlim(-5, 100)
                axs[1][0].set_xlim(-5, 100)
                axs[2][0].set_xlim(-5, 100)
                axs[0][1].set_xlim(-5, 100)
                axs[1][1].set_xlim(-5, 100)
                axs[2][1].set_xlim(-5, 100)
                axs[0][0].set_xticklabels([])
                axs[0][1].set_xticklabels([])
                axs[1][0].set_xticklabels([])
                axs[1][1].set_xticklabels([])
                axs[0][0].grid(linestyle='--')
                axs[1][0].grid(linestyle='--')
                axs[2][0].grid(linestyle='--')
                axs[0][1].grid(linestyle='--')
                axs[1][1].grid(linestyle='--')
                axs[2][1].grid(linestyle='--')
                axs[0][0].legend(fontsize=fontsize)
                axs[1][0].legend(fontsize=fontsize)
                axs[2][0].legend(fontsize=fontsize)
                axs[0][1].legend(fontsize=fontsize)
                axs[1][1].legend(fontsize=fontsize)
                axs[2][1].legend(fontsize=fontsize)
                fig.tight_layout()
                plt.savefig(os.path.join(event_path, 'match_plot_residuals_mode-{}.pdf'.format(x)))

                if match_plot==3:
                    fig, axs = plt.subplots(3, 2, figsize=figsize_match)
                    if TEOBResumS:
                        axs[0][0].plot(t, (wf_res[x]['hp']-wf_sxs[x]['hp'])/wf_sxs[x]['hp'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][0].plot(t, (wf_res[x]['hc']-wf_sxs[x]['hc'])/wf_sxs[x]['hc'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][0].plot(t, ((wf_res[x]['hp']-wf_sxs[x]['hp'])-(wf_res[x]['hc']-wf_sxs[x]['hc']))/(wf_sxs[x]['hp']-wf_sxs[x]['hc']), label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, (wf_res[x]['amp']-wf_sxs[x]['amp'])/wf_sxs[x]['amp'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, (wf_res[x]['phi']-wf_sxs[x]['phi'])/wf_sxs[x]['phi'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, (wf_res[x]['omg']-wf_sxs[x]['omg'])/wf_sxs[x]['omg'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    if pyRing:
                        axs[0][0].plot(t, (wf_pyr[x]['hp']-wf_sxs[x]['hp'])/wf_sxs[x]['hp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[1][0].plot(t, (wf_pyr[x]['hc']-wf_sxs[x]['hc'])/wf_sxs[x]['hc'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[2][0].plot(t, ((wf_pyr[x]['hp']-wf_sxs[x]['hp'])-(wf_pyr[x]['hc']-wf_sxs[x]['hc']))/(wf_sxs[x]['hp']-wf_sxs[x]['hc']), label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, (wf_pyr[x]['amp']-wf_sxs[x]['amp'])/wf_sxs[x]['amp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, (wf_pyr[x]['phi']-wf_sxs[x]['phi'])/wf_sxs[x]['phi'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, (wf_pyr[x]['omg']-wf_sxs[x]['omg'])/wf_sxs[x]['omg'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)

                    fig.suptitle(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), q, M, chi1, chi2, 'SXS:BBH:'+event), fontsize=fontsize_title)
                    axs[0][0].set_ylabel(r'$r(h_{+,\ell m})/M$')
                    axs[1][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                    axs[2][0].set_ylabel(r'$r(h_{+,\ell m}-h_{\times,\ell m})/M$')
                    axs[0][1].set_ylabel(r'$rA_{\ell m}/M$')
                    axs[1][1].set_ylabel(r'$\phi_{\ell m}$')
                    axs[2][1].set_ylabel(r'$M\omega_{\ell m}$')
                    axs[2][0].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                    axs[2][1].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                    axs[0][0].set_xlim(-5, 100)
                    axs[1][0].set_xlim(-5, 100)
                    axs[2][0].set_xlim(-5, 100)
                    axs[0][1].set_xlim(-5, 100)
                    axs[1][1].set_xlim(-5, 100)
                    axs[2][1].set_xlim(-5, 100)
                    axs[0][0].set_xticklabels([])
                    axs[0][1].set_xticklabels([])
                    axs[1][0].set_xticklabels([])
                    axs[1][1].set_xticklabels([])
                    axs[0][0].grid(linestyle='--')
                    axs[1][0].grid(linestyle='--')
                    axs[2][0].grid(linestyle='--')
                    axs[0][1].grid(linestyle='--')
                    axs[1][1].grid(linestyle='--')
                    axs[2][1].grid(linestyle='--')
                    axs[0][0].legend(fontsize=fontsize)
                    axs[1][0].legend(fontsize=fontsize)
                    axs[2][0].legend(fontsize=fontsize)
                    axs[0][1].legend(fontsize=fontsize)
                    axs[1][1].legend(fontsize=fontsize)
                    axs[2][1].legend(fontsize=fontsize)
                    fig.tight_layout()
                    plt.savefig(os.path.join(event_path, event, 'match_plot_residuals_renorm_mode-{}.pdf'.format(x)))

plt.show()
plt.close()