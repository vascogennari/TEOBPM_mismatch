# standard python imports
import os
import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import inv, toeplitz
import matplotlib
from matplotlib import ticker, cm
#matplotlib.use('MACOSX')

# pyRing import
import pyRing.waveform as wf
from pyRing.utils import *

# LAL import
import lal

'''
DOCUMENTATION
The script compares the impact of higher modes (l,m) w.r.t. the fundamental
(2,2) for the TEOBPM template in pyRing.

The results are displayed in plots of the mismatch versus two selected
parameters from:
q    : mass ratio of the system     [adimesional]
af   : BH final spin                [adimesional]
dist : distance from the observer   [Mpc]
iota : inclination of the system    [rad]

The PSD-weighted mismatch is evaluated in the time domain [arXiv:2107.05609v2]
and computed between the two waveforms:
(2,2) vs (2,2)+(l,m), for all the selected HMs (l,m).
The plot also show the SNR needed for the HM to be detected [Phys. Rev. D84,
062003 (2011)], with a detection defined from a threshold bayes factor.

WARNING: this code needs to be run with the time delays of the HMs, otherwise
the waveform with HMs is not physical. Specifically, before running the code,
check that the DeltaT term to be uncommented when computing multipole_start_time,
line 1321 of pyRing/waveform.py, TEOBPM class.

--------------------------------
plot
1   plot each mode separately
2   plot selected modes together
0   show both plots 1 and 2

sing_comb
to analyse only one combination of parameters
0           all combinations
non-zero    see the switch in function variables below

DDD_plot
if plot=1, then select the dimensions of the plots
1   2D plots
2   3D plots
0   both 2D and 3D plots
'''


def mode_output(hp, hc):
    ''' return the complex strain, given hp and hc '''
    amp = np.sqrt(hp**2 + hc**2)
    i   = np.argmax(amp)
    phi = np.unwrap(np.angle(hp - 1j*hc))
    phi = -(phi - phi[i])   # reverse phase sgn to make phase grow
    h   = amp * np.exp(1j*phi)
    return h


def FittingFactorTimeDomain(x, y):
    ''' return the normalized scalar product in time domain
        eqs. 2, 3 of Phys. Rev. D99, 064045 (2019) '''
    if len(x) != len(y):
        raise ValueError('\nWFs need to have the same lenght for computing the scalar product. Exiting...')
    num = np.vdot(y,x)
    den = np.sqrt(np.vdot(x,x)*np.vdot(y,y))
    if den==0:
        raise ValueError('\nModes need to be non-zero for computing the scalar product. Exiting...')
    res  = np.abs(num/den)
    return round(res, 5)  # round to avoid numerical noise


def InvCov(samp_rate, n_points, asd_filepath):
    ''' return the inverse of the covariance matrix,
        given sampling rate and number of points
        of the corresponding time array '''
    df = samp_rate/n_points
    dt = 1/samp_rate
    f = np.fft.rfftfreq(n_points, d=dt)

    freq_file, asd_file = np.loadtxt(asd_filepath, unpack=True)
    psd_file = asd_file**2  # convert ASD to PSD

    f_min_psd = 20.0
    f_max_psd = samp_rate/2
    psd_file  = psd_file[freq_file > f_min_psd]
    freq_file = freq_file[freq_file > f_min_psd]
    psd_file  = psd_file[freq_file < f_max_psd]
    freq_file = freq_file[freq_file < f_max_psd]

    PSD = np.interp(f, freq_file, psd_file)
    ACF = 0.5 * np.real(np.fft.irfft(PSD*df)) * n_points    # eq. 45 of arXiv:2107.05609v2

    Covariance_matrix         = toeplitz(ACF)
    Inverse_Covariance_matrix = inv(Covariance_matrix)

    return Inverse_Covariance_matrix


def compute_FF(h1, h2, InvCov):
    ''' return the fitting-factor (FF) weighted with PSD.
        FF is defined as the maximum of the match (or overlap),
        that in our case coincide with the match by construction.
        the scalar product is in time domain
        eq. 50 of arXiv:2107.05609v2 '''
    num    = np.dot(h1, np.dot(InvCov, h2))
    SNR_h1 = compute_SNR_opt(h1, InvCov)
    SNR_h2 = compute_SNR_opt(h2, InvCov)
    FF     = np.minimum(1, np.abs(num/(SNR_h1*SNR_h2)))   # take the minimun to avoid numerical overflow

    return round(FF, 5), round(SNR_h1, 5), round(SNR_h2, 5)  # round to avoid numerical noise


def compute_SNR_opt(h, InvCov):
    ''' return the optimal SNR in time domain
        eq. 51 of arXiv:2107.05609v2 '''
    SNR_opt = np.sqrt(np.dot(h, np.dot(InvCov, h)))
    return np.abs(SNR_opt)


def SNR_threshold_from_FF(FF, logB_threshold):
    ''' return the threshold SNR for a given fitting-factor,
        having fixed the detection threshold logB_threshold '''
    res = np.sqrt((2*logB_threshold)/(1-FF**2))   # eq. 26 of Phys. Rev. D84, 062003 (2011)
    return res


def variables(flag):
    ''' switch containing the possible
        pairs of parameters '''
    switch = { 1 : (('q'          , 'af')         , (q_range, chi_range)    , ('$\iota$', '$D$')                 , (iota, dist)       , ('$q$', '$a_f$')),
               2 : (('q'          , 'inclination'), (q_range, incl_range)   , ('$\chi_1$', '$\chi_2$', '$D$')    , (chi1, chi2, dist) , ('$q$', '$\iota$ [rad]')), 
               3 : (('q'          , 'distance')   , (q_range, dist_range)   , ('$\chi_1$', '$\chi_2$', '$\iota$'), (chi1, chi2, iota) , ('$q$', '$D$ [Mpc]')),
               4 : (('inclination', 'af')         , (incl_range, chi_range) , ('$q$', '$D$')                     , (q_val, dist)      , ('$\iota$ [rad]', '$a_f$')),
               5 : (('distance'   , 'af')         , (dist_range, chi_range) , ('$q$', '$\iota$')                 , (q_val, iota)      , ('$D$ [Mpc]', '$a_f$')),
               6 : (('distance'   , 'inclination'), (dist_range, incl_range), ('$\chi_1$', '$\chi_2$', '$q$')    , (chi1, chi2, q_val), ('$D$ [Mpc]', '$\iota$ [rad]'))}

    return switch.get(flag)



class Mismatch:
    ''' -------------------------------------------------------------- '''
    ''' build dictionaries with the values of:                         '''
    ''' mismatch, threshold SNR, selected pair of parameters           '''
    ''' -------------------------------------------------------------- '''
    def __init__(self, mm_params, TEOB_params, t, InvCov):
        
        self.x = np.linspace(mm_params['range'][0][0], mm_params['range'][0][1], mm_params['points'])
        self.y = np.linspace(mm_params['range'][1][0], mm_params['range'][1][1], mm_params['points'])

        if (mm_params['label'][1]=='af'):
            self.af     = np.zeros(mm_params['points']*mm_params['points'])
            self.mm_af  = {'{}{}'.format(lm[0], lm[1]): np.zeros((mm_params['points'], mm_params['points']*mm_params['points'])) for lm in TEOB_params['modes']}
            self.snr_af = {'{}{}'.format(lm[0], lm[1]): np.zeros((mm_params['points'], mm_params['points']*mm_params['points'])) for lm in TEOB_params['modes']}

        self.mm  = {'{}{}'.format(lm[0], lm[1]): np.zeros((mm_params['points'], mm_params['points'])) for lm in TEOB_params['modes']}
        self.snr = {'{}{}'.format(lm[0], lm[1]): np.zeros((mm_params['points'], mm_params['points'])) for lm in TEOB_params['modes']}

        for i in range(mm_params['points']):
            if ((mm_params['label'][1]=='af') and (mm_params['label'][0]=='q')):
                p = 0
            for j in range(mm_params['points']):
                if (mm_params['label'][1]=='af'):
                    if (mm_params['label'][0]!='q'):
                        self.q = np.linspace(mm_params['q_range'][0], mm_params['q_range'][1], mm_params['points'])
                        p = 0
                        for k in range(mm_params['points']):
                            for h in range(mm_params['points']):

                                TEOB_params[mm_params['label'][0]] = self.x[i]
                                TEOB_params['chi1'] = self.y[k]
                                TEOB_params['chi2'] = self.y[h]
                                TEOB_params['m1']  = (self.q[j]/(1.+self.q[j])) * TEOB_params['M']
                                TEOB_params['m2']  = (1./(1.+self.q[j])) * TEOB_params['M']

                                # compute TEOB wf with only the fundamental mode
                                TEOBPM_model = wf.TEOBPM(TEOB_params['t0'],
                                                         TEOB_params['m1'],
                                                         TEOB_params['m2'],
                                                         TEOB_params['chi1'],
                                                         TEOB_params['chi2'],
                                                         TEOB_params['phases'],
                                                         TEOB_params['distance'],
                                                         TEOB_params['inclination'],
                                                         TEOB_params['phi'],
                                                         [(2,2)],
                                                         TEOB_params['full-modes'],
                                                         TEOB_params['TGR'],
                                                         TEOB_params['geom'])

                                _, _, _, hp, hc = TEOBPM_model.waveform(t)
                                h_22 = mode_output(hp, hc)
                                self.af[p] = TEOBPM_model.JimenezFortezaRemnantSpin() # final spin

                                # compute TEOB wf with the fundamental and one HM
                                for mode in TEOB_params['modes']:
                                    lm = '{}{}'.format(mode[0], mode[1])

                                    TEOBPM_model = wf.TEOBPM(TEOB_params['t0'],
                                                             TEOB_params['m1'],
                                                             TEOB_params['m2'],
                                                             TEOB_params['chi1'],
                                                             TEOB_params['chi2'],
                                                             TEOB_params['phases'],
                                                             TEOB_params['distance'],
                                                             TEOB_params['inclination'],
                                                             TEOB_params['phi'],
                                                             [(2,2), mode],
                                                             TEOB_params['full-modes'],
                                                             TEOB_params['TGR'],
                                                             TEOB_params['geom'])

                                    _, _, _, hp, hc = TEOBPM_model.waveform(t)
                                    h        = mode_output(hp, hc)
                                    # compute the fitting factor between 22 and 22+lm
                                    FF, _, _ = compute_FF(h_22, h, InvCov)  # scalar product weighted with noise
                                    #FF = FittingFactorTimeDomain(h_22, h)   # scalar product in time domain

                                    self.mm_af[lm][i][p]  = 1 - FF
                                    if (FF==1):
                                        self.snr_af[lm][i][p] = np.nan
                                    else:
                                        self.snr_af[lm][i][p] = SNR_threshold_from_FF(FF, logB_threshold)
                                p += 1
                    else:
                        for k in range(mm_params['points']):

                            TEOB_params[mm_params['label'][0]] = self.x[i]
                            TEOB_params['chi1'] = self.y[j]
                            TEOB_params['chi2'] = self.y[k]
                            TEOB_params['m1']  = (TEOB_params['q']/(1.+TEOB_params['q'])) * TEOB_params['M']
                            TEOB_params['m2']  = (1./(1.+TEOB_params['q'])) * TEOB_params['M']

                            # compute TEOB wf with only the fundamental mode
                            TEOBPM_model = wf.TEOBPM(TEOB_params['t0'],
                                                     TEOB_params['m1'],
                                                     TEOB_params['m2'],
                                                     TEOB_params['chi1'],
                                                     TEOB_params['chi2'],
                                                     TEOB_params['phases'],
                                                     TEOB_params['distance'],
                                                     TEOB_params['inclination'],
                                                     TEOB_params['phi'],
                                                     [(2,2)],
                                                     TEOB_params['full-modes'],
                                                     TEOB_params['TGR'],
                                                     TEOB_params['geom'])

                            _, _, _, hp, hc = TEOBPM_model.waveform(t)
                            h_22 = mode_output(hp, hc)
                            self.af[p] = TEOBPM_model.JimenezFortezaRemnantSpin() # final spin

                            # compute TEOB wf with the fundamental and one HM
                            for mode in TEOB_params['modes']:
                                lm = '{}{}'.format(mode[0], mode[1])

                                TEOBPM_model = wf.TEOBPM(TEOB_params['t0'],
                                                         TEOB_params['m1'],
                                                         TEOB_params['m2'],
                                                         TEOB_params['chi1'],
                                                         TEOB_params['chi2'],
                                                         TEOB_params['phases'],
                                                         TEOB_params['distance'],
                                                         TEOB_params['inclination'],
                                                         TEOB_params['phi'],
                                                         [(2,2), mode],
                                                         TEOB_params['full-modes'],
                                                         TEOB_params['TGR'],
                                                         TEOB_params['geom'])

                                _, _, _, hp, hc = TEOBPM_model.waveform(t)
                                h        = mode_output(hp, hc)
                                # compute the fitting factor between 22 and 22+lm
                                FF, _, _ = compute_FF(h_22, h, InvCov)  # scalar product weighted with noise
                                #FF = FittingFactorTimeDomain(h_22, h)   # scalar product in time domain

                                self.mm_af[lm][i][p]  = 1 - FF
                                if (FF==1):
                                    self.snr_af[lm][i][p] = np.nan
                                else:
                                    self.snr_af[lm][i][p] = SNR_threshold_from_FF(FF, logB_threshold)
                            p += 1
                else:
                    TEOB_params[mm_params['label'][0]] = self.x[i]
                    TEOB_params[mm_params['label'][1]] = self.y[j]

                    TEOB_params['m1']  = (TEOB_params['q']/(1.+TEOB_params['q'])) * TEOB_params['M']
                    TEOB_params['m2']  = (1./(1.+TEOB_params['q'])) * TEOB_params['M']

                    # compute TEOB wf with only the fundamental mode
                    TEOBPM_model = wf.TEOBPM(TEOB_params['t0'],
                                            TEOB_params['m1'],
                                            TEOB_params['m2'],
                                            TEOB_params['chi1'],
                                            TEOB_params['chi2'],
                                            TEOB_params['phases'],
                                            TEOB_params['distance'],
                                            TEOB_params['inclination'],
                                            TEOB_params['phi'],
                                            [(2,2)],
                                            TEOB_params['full-modes'],
                                            TEOB_params['TGR'],
                                            TEOB_params['geom'])

                    _, _, _, hp, hc = TEOBPM_model.waveform(t)
                    h_22 = mode_output(hp, hc)

                    # compute TEOB wf with the fundamental and one HM
                    for mode in TEOB_params['modes']:
                        lm = '{}{}'.format(mode[0], mode[1])

                        TEOBPM_model = wf.TEOBPM(TEOB_params['t0'],
                                                TEOB_params['m1'],
                                                TEOB_params['m2'],
                                                TEOB_params['chi1'],
                                                TEOB_params['chi2'],
                                                TEOB_params['phases'],
                                                TEOB_params['distance'],
                                                TEOB_params['inclination'],
                                                TEOB_params['phi'],
                                                [(2,2), mode],
                                                TEOB_params['full-modes'],
                                                TEOB_params['TGR'],
                                                TEOB_params['geom'])

                        _, _, _, hp, hc = TEOBPM_model.waveform(t)
                        h        = mode_output(hp, hc)
                        # compute the fitting factor between 22 and 22+lm
                        FF, _, _ = compute_FF(h_22, h, InvCov)  # scalar product weighted with noise
                        #FF = FittingFactorTimeDomain(h_22, h)   # scalar product in time domain

                        self.mm[lm][i][j]  = 1 - FF
                        if (FF==1):
                            self.snr[lm][i][j] = np.nan
                        else:
                            self.snr[lm][i][j] = SNR_threshold_from_FF(FF, logB_threshold)

        if (mm_params['label'][1]=='af'):
            for mode in TEOB_params['modes']:
                lm = '{}{}'.format(mode[0], mode[1])

                for i in range(mm_params['points']):
                    self.mm_af[lm][i]  = self.mm_af[lm][i][np.argsort(self.af)]
                    self.mm[lm][i]     = self.mm_af[lm][i][::mm_params['points']]
                    self.snr_af[lm][i] = self.snr_af[lm][i][np.argsort(self.af)]
                    self.snr[lm][i]    = self.snr_af[lm][i][::mm_params['points']]

            self.af = np.sort(self.af)
            self.y = self.af[::mm_params['points']]
    
    def read_params(self):
        return self.mm, self.snr, self.x, self.y


if __name__=='__main__':
# -------------------------------------------------- #
    # select init parameters
    all_modes = 0           # flag to select all available modes
    geom      = 0           # flag to plot in geometric units
    plot      = 1           # see description at the beginning of the file
    sing_comb = 2           # see description at the beginning of the file
    DDD_plot  = 1           # see description at the beginning of the file

    # select the modes
    modes     = [(3,3),(2,1),(3,2),(4,4)]
    #modes     = [(3,3),(2,1)]
    points    = 30          # points in the grid for each variable

    # set ranges and parameter values
    # q, chi are adimensional, distance in [Mpc], iota [rad]
    M     = 100
    q_val = 2
    chi1  = -0.7
    chi2  = 0.0
    dist  = 450
    iota  = 0.7

    q_range    = (1, 3)
    chi_range  = (-0.8, 0.95)
    dist_range = (1e2, 1e3)
    cosi_range = (1, 0)    # cos(iota)
    incl_range = np.arccos(cosi_range)

    # set the tresholds for HM detection
    logB_threshold = 5

    # set ASD filename that will be used to compute the scalar product
    asd_filename = 'ASD_LIGO-P1200087-v18-aLIGO_DESIGN.txt'
# -------------------------------------------------- #

    # utils section
    conv_mass_time = lal.MSUN_SI * lal.G_SI / (lal.C_SI**3)     # mass-time units conversion
    srate  = 4096.0
    T_mass = 30  # end of time array in [M]
    T      = T_mass*conv_mass_time*M
    t_num  = int(srate*T)
    t_start, t_end = 0, T/2.
    t      = np.linspace(t_start, t_end, t_num)
    t_int  = t_end-t_start

    if all_modes:
        modes = [(2,1), (3,3), (4,4)]

    if (geom==1) and not (sing_comb in [0, 1]):
        raise ValueError('\nCannot use geometrical units for combinations including extrinsic parameters. Exiting...')

    # set working directories and paths
    work_path  = os.getcwd()
    if not os.path.exists('TEOBPM_mismatch_HMs_results'):
        os.makedirs('TEOBPM_mismatch_HMs_results')
    plot_path  = os.path.join(work_path, 'TEOBPM_mismatch_HMs_results')

    asd_filepath = 'PSDs/'+asd_filename

    modes_string = ''
    for mode in modes:
        lm = '{}{}'.format(mode[0], mode[1])
        modes_string += '-{}'.format(lm)

    phases = {(2,2):0.0, (2,1):0.0, (3,3):0.0, (3,2):0.0, (4,4):0.0, (4,3):0.0, (4,2):0.0}

    # main core
    for c in range(1,7):    # loop on pairs of parameters

        if sing_comb:
            c = sing_comb
        else:
            print("Computed: {}/{}\033[A".format(c, 6))

        TEOB_params = {'t0'          : 0.0   ,
                    'q'           : q_val ,
                    'M'           : M     ,
                    'm1'          : 0.0   ,
                    'm2'          : 0.0   ,
                    'chi1'        : chi1  ,
                    'chi2'        : chi2  ,
                    'phases'      : phases,
                    'distance'    : dist  ,
                    'inclination' : iota  ,
                    'phi'         : 0.0   ,
                    'modes'       : modes ,
                    'full-modes'  : 0     ,
                    'TGR'         : {}    ,
                    'geom'        : geom  }

        mm_params = {'points' : points,
                    'label'  : variables(c)[0],
                    'range'  : variables(c)[1],
                    'q_range': q_range}

        if ((c==2) or (c==3) or (c==6)):
            figure_names = {'title' : '{}={}, {}={}'.format(variables(c)[2][0], variables(c)[3][0], variables(c)[2][1], variables(c)[3][1])}
        else:
            figure_names = {'title' : '{}={}'.format(variables(c)[2][0], variables(c)[3][0])}
        figure_names['x_label'] = '{}'.format(variables(c)[4][0])
        figure_names['y_label'] = '{}'.format(variables(c)[4][1])

        InvCov_matrix   = InvCov(srate, t_num, asd_filepath)
        mismatch_output = Mismatch(mm_params, TEOB_params, t, InvCov_matrix)
        Z, SNR, x_array, y_array = mismatch_output.read_params()


        ''' ------------- '''
        ''' PLOTS SECTION '''
        ''' ------------- '''
        X, Y = np.meshgrid(x_array, y_array)

        if (plot==0 or plot==1):
            ''' for each mode, plot of the mismatch as function of the
                selected parameters pair. on the x-y plane, plot the
                contourf of the threshold SNR for detection. '''
            for mode in modes:
                lm = '{}{}'.format(mode[0], mode[1])

                if (DDD_plot==0 or DDD_plot==2):  # 3D plot of SNR/logB
                    fig = plt.figure()
                    ax  = plt.axes(projection ='3d')
                    ax.plot_wireframe(X, Y, Z[lm].T, color='black', linewidth=0.5)
                    ctrf = ax.contourf(X, Y, SNR[lm].T, zdir='z', offset=Z[lm].min(), cmap='RdYlBu_r')
                    cbar  = fig.colorbar(ctrf, ax=ax, shrink=0.4, aspect=10)
                    cbar.ax.set_title(r'SNR')
                    
                    ax.set_zlim(Z[lm].min(), Z[lm].max())
                    if geom:
                        ax.set_title('SNR MODE {}'.format(lm))
                    else:
                        ax.set_title('SNR MODE {}  -  {}'.format(lm, figure_names['title']))
                    ax.set_xlabel('{}'.format(figure_names['x_label']))
                    ax.set_ylabel('{}'.format(figure_names['y_label']))
                    ax.set_zlabel(r'$MM$')

                    ax.grid(linestyle='--')
                    fig.tight_layout()
                    plt.savefig(os.path.join(plot_path, 'mismatch_3D_{}_{}-{}.pdf'.format(lm, mm_params['label'][0], mm_params['label'][1])))

                if (DDD_plot==0 or DDD_plot==1):  # 2D plot of SNR/logB
                    fig = plt.figure()
                    ax  = plt.axes()
                    ctrf = ax.contourf(X, Y, SNR[lm].T, locator=ticker.LogLocator(), cmap ="RdYlBu_r", levels=[5,10,15,25,50,100,500,1000])
                    cbar = plt.colorbar(ctrf)

                    if geom:
                        ax.set_title('SNR MODE {}'.format(lm))
                    else:
                        ax.set_title('SNR MODE {}  -  {}'.format(lm, figure_names['title']))
                    ax.set_xlabel('{}'.format(figure_names['x_label']))
                    ax.set_ylabel('{}'.format(figure_names['y_label']))
                    #ax.grid(linestyle='--')
                    fig.tight_layout()
                    plt.savefig(os.path.join(plot_path, 'mismatch_{}_{}-{}.pdf'.format(lm, mm_params['label'][0], mm_params['label'][1])))

        if (plot==0 or plot==2):
            ''' comparison of the mismtch for the selected modes
                as a function of the selected parameters pair '''
            fig = plt.figure()
            ax  = plt.axes(projection ='3d')

            for mode in modes:
                lm = '{}{}'.format(mode[0], mode[1])
                surf = ax.plot_surface(X, Y, Z[lm].T*100,label='{}'.format(lm))
                surf._edgecolors2d = surf._edgecolor3d
                surf._facecolors2d = surf._facecolor3d

            if geom:
                ax.set_title('HIGHER MODES COMPARISON')
            else:
                ax.set_title('HIGHER MODES COMPARISON  -  {}'.format(figure_names['title']))
            ax.set_xlabel('{}'.format(figure_names['x_label']))
            ax.set_ylabel('{}'.format(figure_names['y_label']))
            ax.set_zlabel(r'$MM$%')

            ax.legend()
            ax.grid(linestyle='--')
            fig.tight_layout()
            plt.savefig(os.path.join(plot_path, 'mismatch{}_{}-{}.pdf'.format(modes_string, mm_params['label'][0], mm_params['label'][1])))

        if sing_comb: break
        if geom: break

plt.show()
plt.close()