from librosa import core
from librosa import beat
import numpy as np

def get_beat_sync_spectrums(audio):
    y, sr = core.load(audio, sr=44100)
    tempo, beats = beat.beat_track(y, sr=sr, start_bpm=110, units='time', trim=False)
    framed_dbn = np.concatenate([np.array([0]), beats ])
    band1 = (0, 220)
    band2 = (220, 1760)
    band3 = (1760, sr/2)
    band1list = []
    band2list = []
    band3list = []
    for i in range(1, len(framed_dbn)):
      fft = abs(np.fft.fft(y[int(framed_dbn[i-1]*sr):int(framed_dbn[i]*sr)]))
      freqs = np.fft.fftfreq(len(fft), 1/sr)
      band1list.append(np.sum(fft[np.where(np.logical_and(freqs > band1[0], freqs < band1[1]))]))
      band2list.append(np.sum(fft[np.where(np.logical_and(freqs > band2[0], freqs < band2[1]))]))
      band3list.append(np.sum(fft[np.where(np.logical_and(freqs > band3[0], freqs < band3[1]))]))
    
    band1list = np.array(band1list).transpose()
    band2list = np.array(band2list).transpose()
    band3list = np.array(band3list).transpose()
    return np.vstack([band1list, band2list, band3list])