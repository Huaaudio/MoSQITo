import utils.load as load
import sq_metrics.loudness.loudness_zwst.loudness_zwst as loudness_zwst

# Load the signal
signal, fs = load.load("BP570_scaled.wav")
loudness = loudness_zwst.loudness_zwst(signal, fs, field_type="free")
print(loudness)