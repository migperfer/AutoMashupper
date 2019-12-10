from librosa import beat
from librosa import core
from librosa import feature
import matplotlib.pyplot as plt
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as downbeattrack
from madmom.features.downbeats import RNNDownBeatProcessor as beatrnn
import numpy as np

def get_beat_sync_chroma(audio):
    y, sr = core.load(audio, sr=44100)
    tempo, _ = beat.beat_track(y, sr=sr, start_bpm=110, units='time', trim=False)
    act = beatrnn()(audio)
    beats = downbeattrack(beats_per_bar=[4, 4], fps=100)(act)
    downbeats = beats[beats[:, 1] == 1][:][:, 0]
    framed_dbn = np.concatenate([np.array([0]), downbeats, np.array([len(y)/sr])])
    time = np.arange(len(y)) / sr
    chromas = []
    for i in range(1, len(framed_dbn)):
        chroma = np.mean(feature.chroma_stft(y[int(framed_dbn[i-1]*sr):int(framed_dbn[i]*sr)], sr=sr, ), axis=1)
        chromas.append(chroma)
    chromas = np.array(chromas).transpose()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time, y)
    ax[0].vlines(framed_dbn, -1, 1, colors='r', linestyles='dashdot')
    ax[0].set_xlim(framed_dbn[0], framed_dbn[-1])
    plt.sca(ax[1])
    plt.pcolor(chromas)
    plt.show()
    print(tempo)

    return chromas, downbeats, tempo

if __name__ == '__main__':
    get_beat_sync_chroma("galaxy-synth.wav")