import numpy as np
import pywt
import matplotlib.pyplot as plt

from utils.conversion import ms_to_samples

"""
P-QRS-T Detector implementation based on "Detection of P, QRS, and T Components of ECG
Using Wavelet Transformation" by Abed Al Raoof Bsoul et al.

Implements a qrs detector which uses DWT to extract significant waves within the ECG signal.
Uses Haar wavelet to scan for R peaks and Daubechies wavelet of order 2 to determine the position of P and T waves.
"""
class QRSFeatureReader(object):
    def __init__(self, sample_rate=360, **kwargs):
        # Sample rate of the data
        self._sample_rate = sample_rate
        # Value defining distance to Q and S when searching for P and T to overcome noise peaks.
        self._noise_comp = kwargs.get('noise_comp', 0)
        # The level of wavelet transform to use
        self._dwt_level = kwargs.get('dwt_level', 4)
        # Size of the R correction window
        self._r_corr_window = kwargs.get('r_correction_window', 24) # TODO Must scale with other sample rates
        # Size of the Q and S detection windows
        self._qs_dect_window = int(0.12 * self._sample_rate)
        # Refractory period in samples
        self._ref_time = ms_to_samples(200, sample_rate=sample_rate)

    """
    Starts the detection of the ecg features based on the given mode. Defaults to PQRST.
    Options: R, QRS, PQRST
    """
    def detect(self, ds, mode='pqrst'):
        # Create or clear detected
        ds.p = []
        ds.q = []
        ds.r = []
        ds.s = []
        ds.t = []

        if 'r' in mode:
            self._detect_r_peaks(ds)
            self._correct_r_peaks(ds)

        if 'q' and 's' in mode:
            self._detect_qs_peaks(ds)

        if 'p' and 't' in mode:
            self._detect_pt_peaks(ds)

    """
    Runs R peak detection using DWT and haar wavelet.
    """
    def _detect_r_peaks(self, ds):
        wav = pywt.Wavelet('haar')
        # Multi resolution decomposition of the signal to improve SNR
        coeffs = pywt.swt(ds.data.flatten(), wavelet=wav, level=self._dwt_level)
        det = coeffs[3][1] ** 2
        # Threshold where to define a peak as R
        thr = 1.5 * np.std(det)
        cursor = 0

        while cursor < len(det):
            if det[cursor] >= thr:
                # Add peak as candidate
                ds.r.append(cursor)
                # Skip next samples due to physiological refractory period
                cursor += self._ref_time
            else:
                cursor += 1

    """
    Correct the R peak location from the wavelet based detection.
    """
    def _correct_r_peaks(self, ds):
        for idx, r in enumerate(ds.r):
            # Define left side and right side indices to create centered correction window
            lhw = r - (self._r_corr_window >> 1)
            rhw = r + (self._r_corr_window >> 1)

            # Prevent wrong windows due to out of bounds
            if lhw < 0: lhw = 0
            if rhw >= len(ds.data): rhw = len(ds.data) - 1

            # A subarray centered around the detected peak
            window = ds.data[lhw:rhw]

            # Detect R phase
            w_min = min(window)
            w_max = max(window)

            # R peak is positive, therefore Q an S must be minima
            if abs(w_max) > abs(w_min):
                # Calculate corrected position relative to the window
                r_idx = r + np.argmax(window) - (self._r_corr_window >> 1)

                # Prevent indices out of bounds
                if r_idx < 0: r_idx = 0
                if r_idx >= len(ds.data): r_idx = len(ds.data) - 1

                # Store corrected position and direction of the peak as tuple
                ds.r[idx] = (r_idx, 1)
            else:
                # R peak negative, Q and S as maxima - calculate corrected position relative to the window
                r_idx = r + np.argmin(window) - (self._r_corr_window >> 1)

                # Prevent indices out of bounds
                if r_idx < 0: r_idx = 0
                if r_idx >= len(ds.data): r_idx = len(ds.data) - 1

                # Store corrected position and direction of the peak as tuple
                ds.r[idx] = (r_idx, 0)

    """
    Detects Q and S values of the given ecg wave.
    """
    def _detect_qs_peaks(self, ds):
        for peak, phase in ds.r:
            # Define left side and right side indices to create detection windows
            lh_idx = peak - self._qs_dect_window
            rh_idx = peak + self._qs_dect_window

            # Prevent indices out of bounds
            if lh_idx < 0: lh_idx = 0
            if rh_idx >= len(ds.data): rh_idx = len(ds.data) - 1

            # Two subarrays left and right next to the corrected r peak
            lha = ds.data[lh_idx:peak]
            rha = ds.data[peak:rh_idx]

            # Peak was marked as negative, Q and S must be maxima
            if phase == 0:
                # Find Q and S positions relative to detection window
                q_idx = peak + np.argmax(lha) - self._qs_dect_window
                s_idx = peak + np.argmax(rha)
                # Prevent out of bounds and store position
                ds.q.append(q_idx if q_idx > 0 else 0)
                ds.s.append(s_idx if s_idx < len(ds.data) else len(ds.data) - 1)
            elif phase == 1:
                # Peak was marked as positive, Q and S must be minima
                # Find Q and S positions relative to detection window
                q_idx = peak + np.argmin(lha) - self._qs_dect_window
                s_idx = peak + np.argmin(rha)
                # Prevent out of bounds and store position
                ds.q.append(q_idx if q_idx > 0 else 0)
                ds.s.append(s_idx if s_idx < len(ds.data) else len(ds.data) - 1)

    """
    Detects P and T wave values of the given ecg value using DWT and DB2 wavelets.
    """
    def _detect_pt_peaks(self, ds):
        cycle_count = 0
        self._noise_comp = 20

        while cycle_count < len(ds.r):
            # Create a "cardiac cycle" between two R-R peaks, choose next neighbors if not exist
            if cycle_count == 0:
                cr1 = (ds.r[cycle_count][0] - ((ds.r[cycle_count + 2][0] - ds.r[cycle_count + 1][0]) >> 1))
                cr2 = (ds.r[cycle_count][0] + ds.r[cycle_count + 1][0]) >> 1
            elif cycle_count == len(ds.r) - 1:
                cr1 = (ds.r[cycle_count][0] - ((ds.r[cycle_count][0] - ds.r[cycle_count - 1][0]) >> 1))
                cr2 = (ds.r[cycle_count][0] + ((ds.r[cycle_count - 1][0] - ds.r[cycle_count - 2][0]) >> 1))
            else:
                cr1 = (ds.r[cycle_count][0] - ((ds.r[cycle_count][0] - ds.r[cycle_count - 1][0]) >> 1))
                cr2 = (ds.r[cycle_count][0] + ds.r[cycle_count + 1][0]) >> 1

            # Prevent out of bounds
            if cr1 < 0: cr1 = 1
            if cr2 >= len(ds.data): cr2 = len(ds.data) - 1

            # Get the length of the P and T detection window
            p_win_len = ds.q[cycle_count] - self._noise_comp - cr1
            t_win_len = cr2 - ds.s[cycle_count] - self._noise_comp

            # Ignore P and T detection for first peak if its detection window lies out of bounds
            if p_win_len < 0 or t_win_len < 0:
                cycle_count += 1
                continue

            # Check for an odd length of the P and T subarrays, add an value if true as wavelet transform requires an
            # even amount of data. Create detection subarrays for P and T.
            if p_win_len & 0x1 == 1:
                p_window = ds.data[cr1 - 1:ds.q[cycle_count] - self._noise_comp]
            else:
                p_window = ds.data[cr1:ds.q[cycle_count] - self._noise_comp]

            if t_win_len & 0x1 == 1:
                t_window = ds.data[ds.s[cycle_count] + self._noise_comp:cr2 + 1]
            else:
                t_window = ds.data[ds.s[cycle_count] + self._noise_comp:cr2]

            # Store P position
            ds.p.append(self._calculate_pt_locations('p', ds, cycle_count, p_window))
            # Store T position
            ds.t.append(self._calculate_pt_locations('t', ds, cycle_count, t_window))

            cycle_count += 1

    """
    Searches within the P-T-subarrays for the index with the most energy.
    """
    def _calculate_pt_locations(self, wave, ds, cc, window):
        # The wavelet to use for PT extraction
        wav = pywt.Wavelet('haar')
        # Use wavelet decomposition until level 4 to retrieve energy of the P and T wave
        coeff = pywt.downcoef('d', window.flatten(), wavelet=wav, level=4) ** 2
        # Use std as threshold value
        std = 1.5 * np.std(coeff)
        max_idx = -1

        # Search for the index where the energy is maxed
        for idx, val in enumerate(coeff):
            if val >= std and (max_idx == -1 or coeff[idx] > coeff[max_idx]):
                max_idx = idx

        # Correct index within level 4 detail coefficient due to subsampling
        c_idx = max_idx * 2 ** self._dwt_level

        if wave == 'p':
            # Calculate P location relative to the detection window
            r_idx = ds.q[cc] + c_idx - len(window) - self._noise_comp
        else:
            # Calculate T location relative to the detection window
            r_idx = ds.s[cc] + c_idx + self._noise_comp

        # Prevent out of bounds
        if r_idx < 0: r_idx = 0
        if r_idx >= len(ds.data): r_idx = len(ds.data) - 1

        return r_idx