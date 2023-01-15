# TEOBPM_mismatch

In this repo are stored some scripts to study in time domain the characteristics of various gravitational wave (GW) models.\
Currently, it is possible to compare the numerical relativity (NR) simulations from the [`SXS` catalog](https://data.black-holes.org/waveforms/index.html) with the TEOBPM ringdown model implemented in [`pyRing`](https://git.ligo.org/lscsoft/pyring), and the [`TEOBResumS`](https://bitbucket.org/eob_ihes/teobresums/src/master/) inspiral-merger-ringdown model.\\


### `SXS_mismatch_catalog.py`
Compute the mismatch in time domain between TEOBPM and TEOBResumS with simulations from the SXS catalog.

### `SXS_mismatch_single_event.py`
For the selected SXS simulation, compare different parameters of the waveform between the various models.

### `TEOBPM_mismatch_HMs.py`
Evaluate the contribution of higher modes (HMs) compared to the fundamental mode and assess their detectability as a function of the signal-to-noise ratio (SNR). This procedure is currently available only for the TEOBPM model.

### `sxs_download.py`
Script to autoamatically download the simulations with aligned or anti-aligned simulations from the SXS catalog.
