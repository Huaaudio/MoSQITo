import numpy as np
from scipy import signal
from dataclasses import dataclass
from numba import jit, float64

@dataclass
class ThirdOctaveFilterCoeffs:
    reference: np.ndarray  # Reference filter coefficients
    differences: np.ndarray  # Difference from reference for each band
    gains: np.ndarray  # Filter gains for each stage

def get_filter_coeffs() -> ThirdOctaveFilterCoeffs:
    """Returns the filter coefficients from ISO 532-1 standard"""
    # Reference filter coefficients [3 stages][6 coeffs: b0,b1,b2,a0,a1,a2]
    reference = np.array([
        [1, 2, 1, 1, -2, 1],
        [1, 0, -1, 1, -2, 1],
        [1, -2, 1, 1, -2, 1]
    ])
    
    # Difference coefficients for each band [28 bands][3 stages][6 coeffs]
    # Only showing first few bands for brevity - add all coefficients from ISO standard
    differences = np.array([
    [ [0,0,0,0,-6.70260e-004,6.59453e-004],
      [0,0,0,0,-3.75071e-004,3.61926e-004],
      [0,0,0,0,-3.06523e-004,2.97634e-004] ],

    [ [0,0,0,0,-8.47258e-004,8.30131e-004],
      [0,0,0,0,-4.76448e-004,4.55616e-004],
      [0,0,0,0,-3.88773e-004,3.74685e-004] ],

    [ [0,0,0,0,-1.07210e-003,1.04496e-003],
      [0,0,0,0,-6.06567e-004,5.73553e-004],
      [0,0,0,0,-4.94004e-004,4.71677e-004] ],

    [ [0,0,0,0,-1.35836e-003,1.31535e-003],
      [0,0,0,0,-7.74327e-004,7.22007e-004],
      [0,0,0,0,-6.29154e-004,5.93771e-004] ],

    [ [0,0,0,0,-1.72380e-003,1.65564e-003],
      [0,0,0,0,-9.91780e-004,9.08866e-004],
      [0,0,0,0,-8.03529e-004,7.47455e-004] ],

    [ [0,0,0,0,-2.19188e-003,2.08388e-003],
      [0,0,0,0,-1.27545e-003,1.14406e-003],
      [0,0,0,0,-1.02976e-003,9.40900e-004] ],

    [ [0,0,0,0,-2.79386e-003,2.62274e-003],
      [0,0,0,0,-1.64828e-003,1.44006e-003],
      [0,0,0,0,-1.32520e-003,1.18438e-003] ],

    [ [0,0,0,0,-3.57182e-003,3.30071e-003],
      [0,0,0,0,-2.14252e-003,1.81258e-003],
      [0,0,0,0,-1.71397e-003,1.49082e-003] ],

    [ [0,0,0,0,-4.58305e-003,4.15355e-003],
      [0,0,0,0,-2.80413e-003,2.28135e-003],
      [0,0,0,0,-2.23006e-003,1.87646e-003] ],

    [ [0,0,0,0,-5.90655e-003,5.22622e-003],
      [0,0,0,0,-3.69947e-003,2.87118e-003],
      [0,0,0,0,-2.92205e-003,2.36178e-003] ],

    [ [0,0,0,0,-7.65243e-003,6.57493e-003],
      [0,0,0,0,-4.92540e-003,3.61318e-003],
      [0,0,0,0,-3.86007e-003,2.97240e-003] ],

    [ [0,0,0,0,-1.00023e-002,8.29610e-003],
      [0,0,0,0,-6.63788e-003,4.55999e-003],
      [0,0,0,0,-5.15982e-003,3.75306e-003] ],

    [ [0,0,0,0,-1.31230e-002,1.04220e-002],
      [0,0,0,0,-9.02274e-003,5.73132e-003],
      [0,0,0,0,-6.94543e-003,4.71734e-003] ],

    [ [0,0,0,0,-1.73693e-002,1.30947e-002],
      [0,0,0,0,-1.24176e-002,7.20526e-003],
      [0,0,0,0,-9.46002e-003,5.93145e-003] ],

    [ [0,0,0,0,-2.31934e-002,1.64308e-002],
      [0,0,0,0,-1.73009e-002,9.04761e-003],
      [0,0,0,0,-1.30358e-002,7.44926e-003] ],

    [ [0,0,0,0,-3.13292e-002,2.06370e-002],
      [0,0,0,0,-2.44342e-002,1.13731e-002],
      [0,0,0,0,-1.82108e-002,9.36778e-003] ],

    [ [0,0,0,0,-4.28261e-002,2.59325e-002],
      [0,0,0,0,-3.49619e-002,1.43046e-002],
      [0,0,0,0,-2.57855e-002,1.17912e-002] ],

    [ [0,0,0,0,-5.91733e-002,3.25054e-002],
      [0,0,0,0,-5.06072e-002,1.79513e-002],
      [0,0,0,0,-3.69401e-002,1.48094e-002] ],

    [ [0,0,0,0,-8.26348e-002,4.05894e-002],
      [0,0,0,0,-7.40348e-002,2.24476e-002],
      [0,0,0,0,-5.34977e-002,1.85371e-002] ],

    [ [0,0,0,0,-1.17018e-001,5.08116e-002],
      [0,0,0,0,-1.09516e-001,2.81387e-002],
      [0,0,0,0,-7.85097e-002,2.32872e-002] ],

    [ [0,0,0,0,-1.67714e-001,6.37872e-002],
      [0,0,0,0,-1.63378e-001,3.53729e-002],
      [0,0,0,0,-1.16419e-001,2.93723e-002] ],

    [ [0,0,0,0,-2.42528e-001,7.98576e-002],
      [0,0,0,0,-2.45161e-001,4.43370e-002],
      [0,0,0,0,-1.73972e-001,3.70015e-002] ],

    [ [0,0,0,0,-3.53142e-001,9.96330e-002],
      [0,0,0,0,-3.69163e-001,5.53535e-002],
      [0,0,0,0,-2.61399e-001,4.65428e-002] ],

    [ [0,0,0,0,-5.16316e-001,1.24177e-001],
      [0,0,0,0,-5.55473e-001,6.89403e-002],
      [0,0,0,0,-3.93998e-001,5.86715e-002] ],

    [ [0,0,0,0,-7.56635e-001,1.55023e-001],
      [0,0,0,0,-8.34281e-001,8.58123e-002],
      [0,0,0,0,-5.94547e-001,7.43960e-002] ],

    [ [0,0,0,0,-1.10165e+000,1.91713e-001],
      [0,0,0,0,-1.23939e+000,1.05243e-001],
      [0,0,0,0,-8.91666e-001,9.40354e-002] ],

    [ [0,0,0,0,-1.58477e+000,2.39049e-001],
      [0,0,0,0,-1.80505e+000,1.28794e-001],
      [0,0,0,0,-1.32500e+000,1.21333e-001] ],

    [ [0,0,0,0,-2.50630e+000,1.42308e-001],
      [0,0,0,0,-2.19464e+000,2.76470e-001],
      [0,0,0,0,-1.90231e+000,1.47304e-001] ],
    ])
    
    # Filter gains [28 bands][3 stages]
    gains = np.array([
        [4.30764e-011, 1, 1],
        [8.59340e-011,1,1],
        [1.71424e-010,1,1],
        [3.41944e-010,1,1],
        [6.82035e-010,1,1],
        [1.36026e-009,1,1],
        [2.71261e-009,1,1],
        [5.40870e-009,1,1],
        [1.07826e-008,1,1],
        [2.14910e-008,1,1],
        [4.28228e-008,1,1],
        [8.54316e-008,1,1],
        [1.70009e-007,1,1],
        [3.38215e-007,1,1],
        [6.71990e-007,1,1],
        [1.33531e-006,1,1],
        [2.65172e-006,1,1],
        [5.25477e-006,1,1],
        [1.03780e-005,1,1],
        [2.04870e-005,1,1],
        [4.05198e-005,1,1],
        [7.97914e-005,1,1],
        [1.56511e-004,1,1],
        [3.04954e-004,1,1],
        [5.99157e-004,1,1],
        [1.16544e-003,1,1],
        [2.27488e-003,1,1],
        [3.91006e-003,1,1]
    ])
    
    return ThirdOctaveFilterCoeffs(reference, differences, gains)

@jit(float64[:](float64[:], float64[:], float64))
def second_order_filter(x: np.ndarray, coeffs: np.ndarray, gain: float) -> np.ndarray:
  """Implements a single second-order filter stage using scipy.signal.lfilter

  Args:
      x: Input signal
      coeffs: Filter coefficients [b0,b1,b2,a0,a1,a2]
      gain: Filter gain

  Returns:
      Filtered signal
  """
  y = np.zeros_like(x)
  wn1 = 0.0
  wn2 = 0.0

  for n in range(len(x)):
    # Exactly matching the ISO C implementation
    wn0 = x[n]*gain - coeffs[4]*wn1 - coeffs[5]*wn2
    y[n] = coeffs[0]*wn0 + coeffs[1]*wn1 + coeffs[2]*wn2
    wn2 = wn1
    wn1 = wn0
  return y

def third_octave_filter(signal: np.ndarray, fs: float, band_index: int) -> float:
    """Filter signal through one third-octave band filter
    
    Args:
        signal: Input time signal
        fs: Sampling frequency (must be 48kHz)
        band_index: Index of the third-octave band (0-27)
    
    Returns:
        RMS level in the band
    """    
    # Get filter coefficients
    coeffs = get_filter_coeffs()
    # Apply three filter stages in cascade
    x = signal.copy()
    for stage in range(3):
        # Combine reference and difference coefficients
        stage_coeffs = coeffs.reference[stage] - coeffs.differences[band_index][stage]
        stage_gain = coeffs.gains[band_index][stage]
        # Apply filter stage
        x = second_order_filter(x, stage_coeffs, stage_gain)
    
    # Return RMS level
    return np.sqrt(np.mean(x**2))

def _n_oct_time_filter(sig: np.ndarray, fs: float, fc: float, alpha: float, N: int = 3) -> float:
    """Updated version using ISO 532-1 filters
    
    Args:
        sig: Time signal
        fs: Sampling frequency (must be 48kHz)
        fc: Center frequency of the band
        alpha: Band edge ratio (ignored - using ISO filters)
        N: Filter order (ignored - using ISO filters)
    
    Returns:
        RMS level in the band
    """
    # Map center frequency to band index
    # Center frequencies follow: f_c = 1000 * 10^((i-16)/10) Hz where i is 0 to 27
    if fc <= 0:
        raise ValueError("Center frequency must be positive")
        
    band_index = round(10 * np.log10(fc/1000) + 16)
    if band_index < 0 or band_index > 27:
        raise ValueError("Center frequency outside valid range for ISO 532-1")
    
    return third_octave_filter(sig, fs, band_index)

if __name__ == "__main__":
    pass
