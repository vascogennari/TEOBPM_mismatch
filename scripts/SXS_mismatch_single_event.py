# Standard python imports
import numpy as np, matplotlib.pyplot as plt
import os

# WFs imports
import sxs                      # SXS
import EOBRun_module            # TEOBResumS
import pyRing.waveform as wf    # pyRing
from pyRing.utils import *

# LVC-specific imports
from lalinference.imrtgr.nrutils import *
import lal

'''
DOCUMENTATION
The script compares the match of TEOBPM w.r.t the NR wfs from the SXS catalog.
Namely, there can be compared both the TEOBResumS and pyRing
implementations of the TEOBPM model.

The output plots show the comparison of the two polarizations, the amplitude,
phase and instantaneous frequency for the selected modes. There is also the
option to show the residuals of the above quantities.

The pyRing wf is evaluated on the same time array given by SXS, while TEOBResumS
automatically generates his own time array. For the residuals, the TEOBResumS wf
is linearly interpolated.

SXS catalog:
https://data.black-holes.org/waveforms/catalog.html

WARNING: for the moment, to correctly run the code you need to manually deselect
the time delays for the HMs from pyRing. Specifically, in the TEOBPM class
in pyRing/waveform.py, the DeltaT term should be commented in the computation
of multipole_start_time (line 1321).
'''


def modes_filt(modes, NR_modes):
    ''' return an array of len(w.LM), with False if the
        mode has not been selected, True otherwise '''
    modes= np.array(modes)
    filt = np.full(len(NR_modes), False)
    i = 0
    for NR_mode in NR_modes:
        for mode in modes:
            if (str(NR_mode) == str(mode)):
                filt[i] = True
        i += 1
    return filt


def modes_to_k(modes):
    ''' return a list of modes using the labeling
        convention of TEOBResumS '''
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]


def mode_output(hp, hc, t, phase_sign=True):
    ''' return a list of arrays containing the
        important quantities of one single mode '''
    amp = np.sqrt(hp**2 + hc**2)
    i   = np.argmax(amp)
    phi = np.unwrap(np.angle(hp - 1j*hc))
    if phase_sign: 
        phi = -(phi - phi[i])   # reverse phase sign to make phase grow (SXS and pyRing)
    else:
        phi = phi - phi[i]      # used for TEOBResumS
    h   = amp * np.exp(1j*phi)
    hp  = np.real(h)
    hc  = -np.imag(h)   # standard convention h = h+ - ihx
    omg = np.zeros_like(phi)
    omg[1:] = np.diff(phi)/np.diff(t)
    return [h, hp, hc, amp, phi, omg]


def rd_cut(h, idx):
    ''' cut the array to select ringdown '''
    return h[idx:]



class SXS_mismatch:
    ''' ------------------------------------------------------------------ '''
    ''' build dictionary with wf quantities for SXS, TEOBResmuS and pyRing '''
    ''' - wf_imr_'model-name' contain the full IMR waveform                '''
    ''' - wf_'model-name' contain only the ringdown part                   '''
    ''' ------------------------------------------------------------------ '''
    def __init__(self, event, flag_params, modes):

        # download wf
        self.extrapolation_order = 2     # 2 is suggested for ringdown
        w = sxs.load(event + '/Lev/rhOverM', extrapolation_order=self.extrapolation_order, download=True)
        metadata = sxs.load(event + '/Lev/metadata.json', download=True)

        # check for spin consistency: chi_1,2 need to be aligned or anti-aligned
        precession = 1e-5
        if ((np.abs(metadata.reference_dimensionless_spin1[0])>precession) or (np.abs(metadata.reference_dimensionless_spin1[1])>precession) or (np.abs(metadata.reference_dimensionless_spin2[0])>precession) or (np.abs(metadata.reference_dimensionless_spin2[1])>precession)):
            raise ValueError('The selected SXS simulation has non-aligned spins. Exiting...')

        # event params
        self.q    = metadata.reference_mass_ratio
        self.m1   = metadata.reference_mass1
        self.m2   = metadata.reference_mass2
        self.chi1 = metadata.reference_dimensionless_spin1[2]   # take the z-component
        self.chi2 = metadata.reference_dimensionless_spin2[2]

        self.M  = self.m1+self.m2
        self.q  = self.m1/self.m2
        self.nu = self.q/((1+self.q)**2)

        # cut the NR noise at the edges
        rd_end = 50
        idx_ini = w.index_closest_to(metadata.reference_time)
        idx_end = w.index_closest_to(w.max_norm_time() + rd_end)
        w = w[idx_ini:idx_end]

        # select hp and hc
        hp_sxs_lm = np.transpose(w.data.real)
        hc_sxs_lm = -np.transpose(w.data.imag)  # minus sign for convention h+-ihx
        mode_filt = modes_filt(modes, w.LM)
        LM_filt   = w.LM[mode_filt]
        hp_sxs_lm = hp_sxs_lm[mode_filt]  # filter the non selected modes
        hc_sxs_lm = hc_sxs_lm[mode_filt]
        t_sxs = w.t

        dict_names_imr = ['h', 'hp', 'hc', 'amp', 'phi', 'omg']
        dict_names     = ['h', 'hp', 'hc', 'amp', 'phi', 'omg', 't']
        self.flag_params = flag_params
        self.event = event

        dist = 450  # distance [Mpc]
        iota = 0.0  # inclination [rad]
        conv_mass_time = lal.MSUN_SI * lal.G_SI / (lal.C_SI**3) # mass-time units conversion

        # SXS
        # -----------------------
        # init two dictionaries with all the wf informations for IMR/RD
        self.wf_imr_sxs = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names_imr} for x in modes}   # IMR
        self.wf_sxs     = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names} for x in modes}    # RD
        self.wf_imr_sxs['fm'] = {y: np.empty(0) for y in dict_names_imr}
        self.wf_imr_sxs['t']  = np.empty(0)

        hp_sxs_fm = np.zeros(len(t_sxs))
        hc_sxs_fm = np.zeros(len(t_sxs))

        for mode in range(len(LM_filt)):
            lm = '{}{}'.format(LM_filt[mode][0], LM_filt[mode][1])
            # IMR
            mode_out = mode_output(hp_sxs_lm[mode], hc_sxs_lm[mode], t_sxs, phase_sign=True)
            for i in range(len(dict_names_imr)):
                self.wf_imr_sxs[lm][dict_names_imr[i]] = mode_out[i]
            # RD
            i_sxs = np.argmax(np.sqrt(hp_sxs_lm[mode]**2 + hc_sxs_lm[mode]**2))  # merger time index as peak of the mode
            t_sxs = t_sxs - t_sxs[i_sxs]
            mode_out.append(t_sxs)
            for x in range(len(mode_out)):
                mode_out[x] = rd_cut(mode_out[x], i_sxs)
            for i in range(len(dict_names)):
                self.wf_sxs[lm][dict_names[i]] = mode_out[i]

            hp_sxs_fm += hp_sxs_lm[mode]
            hc_sxs_fm += hc_sxs_lm[mode]

        mode_out_fm = mode_output(hp_sxs_fm, hc_sxs_fm, t_sxs, phase_sign=True)
        for i in range(len(dict_names_imr)):
            self.wf_imr_sxs['fm'][dict_names_imr[i]] = mode_out_fm[i]
        self.wf_imr_sxs['t'] = t_sxs

        # TEOBResumS
        # -----------------------
        if flag_params['TEOBResumS']:
            k = modes_to_k(modes)
            '''if SXS is selected, input values are
            taken from the selected simulation'''
            pars = {
                'M'                  : self.M,
                'q'                  : self.q,
                'Lambda1'            : 0., # <---- do not change
                'Lambda2'            : 0., # <---- do not change
                'chi1'               : self.chi1,
                'chi2'               : self.chi2,
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
            t_res, _, _, hlm_res, _ = EOBRun_module.EOBRunPy(pars)

            for mode in hlm_res:
                hlm_res['{}'.format(mode)][0] * self.nu  # TEOBResumS wf in geom units is (r h)/(M nu). need to multiply for nu

            # init two dictionaries with all the wf informations for IMR/RD
            self.wf_imr_res = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names_imr} for x in modes}   # IMR
            self.wf_res     = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names} for x in modes}    # RD
            self.wf_imr_res['fm'] = {y: np.empty(0) for y in dict_names_imr}
            self.wf_imr_res['t']  = np.empty(0)

            hp_res_fm = np.zeros(len(t_res))
            hc_res_fm = np.zeros(len(t_res))

            for (j,mode) in enumerate(k):
                lm = '{}{}'.format(modes[j][0], modes[j][1])

                amp_res_lm = hlm_res['{}'.format(mode)][0] * self.nu # TEOBResumS wf in geom units is (r h)/(M nu). need to multiply for nu
                phi_res_lm = hlm_res['{}'.format(mode)][1]
                h_res_lm   = amp_res_lm * np.exp(1j*phi_res_lm)
                hp_res_lm  = np.real(h_res_lm)
                hc_res_lm  = -np.imag(h_res_lm)

                # IMR
                mode_out = mode_output(hp_res_lm, hc_res_lm, t_res, phase_sign=False)
                for i in range(len(dict_names_imr)):
                    self.wf_imr_res[lm][dict_names_imr[i]] = mode_out[i]
                # RD
                i_res = np.argmax(np.sqrt(mode_out[0]**2 + mode_out[1]**2))  # RD start-time index as peak of the mode
                t_res_rd = t_res-t_res[i_res]
                mode_out.append(t_res_rd)
                for x in range(len(mode_out)):
                    mode_out[x] = rd_cut(mode_out[x], i_res)
                for i in range(len(dict_names)):
                    self.wf_res[lm][dict_names[i]] = mode_out[i]

                hp_res_fm += hp_res_lm
                hc_res_fm += hc_res_lm

            mode_out_fm = mode_output(hp_res_fm, hc_res_fm, t_res, phase_sign=False)
            for i in range(len(dict_names_imr)):
                self.wf_imr_res['fm'][dict_names_imr[i]] = mode_out_fm[i]
            self.wf_imr_res['t'] = t_res

            # interpolate TEOBResumS waveform wrt SXS time
            if flag_params['interp']:
                for mode in modes:
                    lm = '{}{}'.format(mode[0], mode[1])
                    for i in range(len(dict_names)):
                        self.wf_res[lm][dict_names[i]] = np.interp(self.wf_sxs[lm]['t'], self.wf_res[lm]['t'], self.wf_res[lm][dict_names[i]])

        # pyRing
        # -----------------------
        '''WARNING: for correct working, need to comment DeltaT in
            the starting time of the modes (waveform.pyx)'''
        if flag_params['pyRing']:
            # init dictionary with all the wf informations for RD
            self.wf_pyr  = {'{}{}'.format(x[0], x[1]): {y: np.empty(0) for y in dict_names} for x in modes}    # RD only

            for mode in modes:
                lm = '{}{}'.format(mode[0], mode[1])
                t = self.wf_sxs[lm]['t']
                phi_0 = self.wf_sxs[lm]['phi'][0]   # set initial phase as the SXS wf phase

                '''if SXS is selected, input values are
                taken from the selected simulation'''
                TEOBPM_model = wf.TEOBPM(0.0,   # <---- start time, do not change
                                         self.m1,
                                         self.m2,
                                         self.chi1,
                                         self.chi2,
                                         {mode: phi_0},
                                         dist,
                                         iota,
                                         0.0,
                                         [mode],
                                         {},
                                         1)     # <---- geometric units, do not change
            
                _, _, _, hp_pyr, hc_pyr = TEOBPM_model.waveform(t)
                self.af = TEOBPM_model.JimenezFortezaRemnantSpin() # final spin

                mode_out = mode_output(hp_pyr, hc_pyr, t, phase_sign=True)
                mode_out.append(t)
                for i in range(len(dict_names)):
                    self.wf_pyr[lm][dict_names[i]] = mode_out[i]

                self.wf_pyr[lm]['omg'][0] = self.wf_pyr[lm]['omg'][1]   #FIXME: this is a temporary hard-fix to avoid the first point to be zero


    def return_final_spin(self):
        return self.af

    def return_event_params(self):
        event_params = {'name': self.event,
                        'extr_order': self.extrapolation_order,
                        'm1':   self.m1,
                        'm2':   self.m2,
                        'chi1': self.chi1,
                        'chi2': self.chi2,
                        'q':    self.q,
                        'M':    self.M,
                        'nu':   self.nu}
        return event_params

    def return_dictionaries(self):

        output_dictionaries = {'SXS': {'IMR': self.wf_imr_sxs, 'RD': self.wf_sxs}}

        if self.flag_params['TEOBResumS']:
            output_dictionaries['TEOBResumS'] = {'IMR': self.wf_imr_res, 'RD': self.wf_res}
        if self.flag_params['pyRing']:
            output_dictionaries['pyRing'] = {'RD': self.wf_pyr}

        return output_dictionaries



if __name__=='__main__':
# -------------------------------------------------- #
    # select init parameters
    event  = '0305'     # name of the SXS simulation:

    # select the modes (if all_modes flag is off)
    modes = [(2,2),(3,3),(2,1),(3,2)]

    # FLAGS
    SXS        = 1      # use SXS <---- do not change, for the moment
    TEOBResumS = 0      # use TEOBResumS
    pyRing     = 1      # use pyRing

    IMR_plots  = 0      # plots of full IMR waveform (only SXS, TEOBResumS). if IMR_plots=2, plot only RD
    match_plot = 1      # match plots of the selected models for RD. if match_plot=2, also plot the residuals

    all_modes  = 1      # select all available modes for the analysis
    interp     = 1      # interpolate TEOBResumS wrt SXS time

    amp_omg    = 1      # show amplitude and frequency for IMR_plots
# -------------------------------------------------- #

    if interp:  # if TEOBResumS is not selected, cannot interpolate TEOBResumS
        if not TEOBResumS:
            interp = 0
    
    event_name = 'SXS:BBH:' + event
    config_flags = {'SXS':        SXS,
                    'TEOBResumS': TEOBResumS,
                    'pyRing':     pyRing,
                    'interp':     interp}
    if all_modes:   # available modes
        modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3), (4,2), (5,5)]

    SXS_mismatch_output = SXS_mismatch(event_name, config_flags, modes)
    output_dictionaries = SXS_mismatch_output.return_dictionaries()
    event_params = SXS_mismatch_output.return_event_params()

    wf_imr_sxs = output_dictionaries['SXS']['IMR']
    wf_sxs     = output_dictionaries['SXS']['RD']
    if TEOBResumS:
        wf_imr_res = output_dictionaries['TEOBResumS']['IMR']
        wf_res     = output_dictionaries['TEOBResumS']['RD']
    if pyRing:
        wf_pyr     = output_dictionaries['pyRing']['RD']


    ''' ------------- '''
    ''' PLOTS SECTION '''
    ''' ------------- '''
    # set working directories and paths
    work_path  = os.getcwd()
    if not os.path.exists('SXS_mismatch_single_event_results'):
        os.makedirs('SXS_mismatch_single_event_results')
    plot_path  = os.path.join(work_path, 'SXS_mismatch_single_event_results')
    dir_name = event + '__q({2:.{0}f})_a({3:.{1}f},{4:.{1}f})'.format(1, 2, event_params['q'], event_params['chi1'], event_params['chi2'])
    if not os.path.exists(os.path.join(plot_path, dir_name)):
        os.makedirs(os.path.join(plot_path, dir_name))
    event_path = os.path.join(plot_path, dir_name)

    # save input parameters in txt file
    with open(os.path.join(event_path, '0_input_parameters.txt'), 'w') as file:
        file.write('The SXS simulation has the following parameters:\n\n')
        for par in event_params:
            file.write('{} = {}\n'.format(par, event_params[par]))
        file.write('\n\nMasses and spins are evaluated in the reference frame.\n')
        file.write('Resolution should be the maximum aivailable, check in metadata if interested.')

    # set plots dimensions
    figsize_imr    = (10,5)
    figsize_match  = (12,7)
    fontsize       = 7
    fontsize_title = 15
    rd_ini = -5
    rd_end = 50


    if IMR_plots:
        ''' FULL IMR WAVEFORM
            plots of IMR waveform for SXS and TEOBResmuS
            if IMR_plots=2 plot only the ringdown part '''
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
                        plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
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
                        plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
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
                    plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
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
                    plt.title(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
                    plt.xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                    plt.ylabel(r'$r(h_{+,\ell m})/M$')
                    plt.grid(linestyle='--')
                    plt.legend(fontsize=fontsize)
                    plt.tight_layout()
                    plt.savefig(os.path.join(event_path, 'RD_TEOBResumS_mode-{}.pdf'.format(x)))

    if match_plot:
        ''' MATCH PLOTS
            match plots for SXS, TEOBResmuS and pyRing
            if match_plot=1 also plot the residuals:
            residuals are computed wrt SXS waveform '''
        for mode in modes:
            x = '{}{}'.format(mode[0], mode[1])
            t = wf_sxs[x]['t']  # set SXS time as global time-array

            if SXS:
                # plots of (h+, hx, h+-hx, amp, phi, omg) for the selected modes
                fig, axs = plt.subplots(3, 2, figsize=figsize_match)
                if SXS:
                    axs[0][0].plot(t, wf_sxs[x]['hp'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                    axs[1][0].plot(t, wf_sxs[x]['hc'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                    axs[2][0].plot(wf_sxs[x]['hp'], wf_sxs[x]['hc'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                    axs[0][1].plot(t, wf_sxs[x]['amp'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                    axs[1][1].plot(t, wf_sxs[x]['phi'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                    axs[2][1].plot(t, wf_sxs[x]['omg'], label=r'SXS', c='#DC183B', linestyle='-', linewidth=1)
                if TEOBResumS:
                    if interp:
                        axs[0][0].plot(t, wf_res[x]['hp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][0].plot(t, wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][0].plot(wf_res[x]['hp'], wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, wf_res[x]['amp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, wf_res[x]['phi'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, wf_res[x]['omg'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                    else:
                        axs[0][0].plot(wf_res[x]['t'], wf_res[x]['hp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][0].plot(wf_res[x]['t'], wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][0].plot(wf_res[x]['hp'], wf_res[x]['hc'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[0][1].plot(wf_res[x]['t'], wf_res[x]['amp'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][1].plot(wf_res[x]['t'], wf_res[x]['phi'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][1].plot(wf_res[x]['t'], wf_res[x]['omg'], label=r'TEOBResumS', c='#1471c0', linestyle='-', linewidth=1)
                if pyRing:
                    axs[0][0].plot(t, wf_pyr[x]['hp'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                    axs[1][0].plot(t, wf_pyr[x]['hc'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                    axs[2][0].plot(wf_pyr[x]['hp'], wf_pyr[x]['hc'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                    axs[0][1].plot(t, wf_pyr[x]['amp'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                    axs[1][1].plot(t, wf_pyr[x]['phi'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)
                    axs[2][1].plot(t, wf_pyr[x]['omg'], label=r'pyRing', c='#426A0F', linestyle='-', linewidth=1)

                fig.suptitle(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
                axs[0][0].set_ylabel(r'$r(h_{+,\ell m})/M$')
                axs[1][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                axs[2][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                axs[0][1].set_ylabel(r'$rA_{\ell m}/M$')
                axs[1][1].set_ylabel(r'$\phi_{\ell m}$')
                axs[2][1].set_ylabel(r'$M\omega_{\ell m}$')
                axs[2][0].set_xlabel(r'$r(h_{+,\ell m})/M$')
                axs[2][1].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                axs[0][0].set_xlim(rd_ini, rd_end)
                axs[1][0].set_xlim(rd_ini, rd_end)
                #axs[2][0].set_xlim(rd_ini, rd_end)
                axs[0][1].set_xlim(rd_ini, rd_end)
                axs[1][1].set_xlim(rd_ini, rd_end)
                axs[2][1].set_xlim(rd_ini, rd_end)
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
                        axs[2][0].plot((wf_res[x]['hp']-wf_sxs[x]['hp']), (wf_res[x]['hc']-wf_sxs[x]['hc']), label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, wf_res[x]['amp']-wf_sxs[x]['amp'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, wf_res[x]['phi']-wf_sxs[x]['phi'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, wf_res[x]['omg']-wf_sxs[x]['omg'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    if pyRing:
                        axs[0][0].plot(t, wf_pyr[x]['hp']-wf_sxs[x]['hp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[1][0].plot(t, wf_pyr[x]['hc']-wf_sxs[x]['hc'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[2][0].plot((wf_pyr[x]['hp']-wf_sxs[x]['hp']), (wf_pyr[x]['hc']-wf_sxs[x]['hc']), label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, wf_pyr[x]['amp']-wf_sxs[x]['amp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, wf_pyr[x]['phi']-wf_sxs[x]['phi'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, wf_pyr[x]['omg']-wf_sxs[x]['omg'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)

                    fig.suptitle(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
                    axs[0][0].set_ylabel(r'$r(h_{+,\ell m})/M$')
                    axs[1][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                    axs[2][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                    axs[0][1].set_ylabel(r'$rA_{\ell m}/M$')
                    axs[1][1].set_ylabel(r'$\phi_{\ell m}$')
                    axs[2][1].set_ylabel(r'$M\omega_{\ell m}$')
                    axs[2][0].set_xlabel(r'$r(h_{+,\ell m})/M$')
                    axs[2][1].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                    axs[0][0].set_xlim(rd_ini, rd_end)
                    axs[1][0].set_xlim(rd_ini, rd_end)
                    #axs[2][0].set_xlim(rd_ini, rd_end)
                    axs[0][1].set_xlim(rd_ini, rd_end)
                    axs[1][1].set_xlim(rd_ini, rd_end)
                    axs[2][1].set_xlim(rd_ini, rd_end)
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
                        axs[2][0].plot((wf_res[x]['hp']-wf_sxs[x]['hp'])/(wf_sxs[x]['hp']), (wf_res[x]['hc']-wf_sxs[x]['hc'])/(wf_sxs[x]['hc']), label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, (wf_res[x]['amp']-wf_sxs[x]['amp'])/wf_sxs[x]['amp'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, (wf_res[x]['phi']-wf_sxs[x]['phi'])/wf_sxs[x]['phi'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, (wf_res[x]['omg']-wf_sxs[x]['omg'])/wf_sxs[x]['omg'], label=r'TEOBResumS-SXS', c='#1471c0', linestyle='-', linewidth=1)
                    if pyRing:
                        axs[0][0].plot(t, (wf_pyr[x]['hp']-wf_sxs[x]['hp'])/wf_sxs[x]['hp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[1][0].plot(t, (wf_pyr[x]['hc']-wf_sxs[x]['hc'])/wf_sxs[x]['hc'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[2][0].plot((wf_pyr[x]['hp']-wf_sxs[x]['hp'])/(wf_sxs[x]['hp']) , (wf_pyr[x]['hc']-wf_sxs[x]['hc'])/(wf_sxs[x]['hc']), label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[0][1].plot(t, (wf_pyr[x]['amp']-wf_sxs[x]['amp'])/wf_sxs[x]['amp'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[1][1].plot(t, (wf_pyr[x]['phi']-wf_sxs[x]['phi'])/wf_sxs[x]['phi'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)
                        axs[2][1].plot(t, (wf_pyr[x]['omg']-wf_sxs[x]['omg'])/wf_sxs[x]['omg'], label=r'pyRing-SXS', c='#426A0F', linestyle='-', linewidth=1)

                    fig.suptitle(r'MODE {2}  -  {7}: q={3:.{1}f}, M={4:.{1}f}, chi1={5:.{1}f}, chi2={6:.{1}f}'.format(0, 1, str(mode), event_params['q'], event_params['M'], event_params['chi1'], event_params['chi2'], 'SXS:BBH:'+event), fontsize=fontsize_title)
                    axs[0][0].set_ylabel(r'$r(h_{+,\ell m})/M$')
                    axs[1][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                    axs[2][0].set_ylabel(r'$r(h_{\times,\ell m})/M$')
                    axs[0][1].set_ylabel(r'$rA_{\ell m}/M$')
                    axs[1][1].set_ylabel(r'$\phi_{\ell m}$')
                    axs[2][1].set_ylabel(r'$M\omega_{\ell m}$')
                    axs[2][0].set_xlabel(r'$r(h_{+,\ell m})/M$')
                    axs[2][1].set_xlabel(r'$t-t_{\rm mrg}$ $[M]$')
                    axs[0][0].set_xlim(rd_ini, rd_end)
                    axs[1][0].set_xlim(rd_ini, rd_end)
                    #axs[2][0].set_xlim(rd_ini, rd_end)
                    axs[0][1].set_xlim(rd_ini, rd_end)
                    axs[1][1].set_xlim(rd_ini, rd_end)
                    axs[2][1].set_xlim(rd_ini, rd_end)
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
                    plt.savefig(os.path.join(event_path, 'match_plot_residuals_renorm_mode-{}.pdf'.format(x)))

    plt.show()
    plt.close()