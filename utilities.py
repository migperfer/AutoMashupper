from librosa import beat
import numpy as np

def rotate_audio(audio, sr, n_beats):
  tempo, _ = beat.beat_track(audio, sr=sr, start_bpm=110, units='time', trim=False)
  samples_rotation = tempo * sr
  n_rotations = int(samples_rotation * n_beats)
  return np.roll(audio, n_rotations)
