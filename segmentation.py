from librosa import beat
from librosa import core
from librosa import feature
import matplotlib.pyplot as plt
import numpy as np

def get_beat_sync_chroma(audio):
    y, sr = core.load(audio, sr=44100)
    tempo, beats = beat.beat_track(y, sr=sr, start_bpm=110, units='time', trim=False)
    time = np.arange(len(y))/sr
    beats = np.concatenate([np.array([0]), beats, np.array([len(y) / sr])])
    chromas = []
    for i in range(1, len(beats)):
        chroma = np.mean(feature.chroma_stft(y[int(beats[i-1]*sr):int(beats[i]*sr)], sr=sr, ), axis=1)
        chromas.append(chroma)
    chromas = np.array(chromas).transpose()
    return chromas, beats, tempo

if __name__ == '__main__':
    get_beat_sync_chroma("galaxy-synth.wav")