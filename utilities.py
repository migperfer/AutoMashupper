from librosa import beat, effects, core, output
import numpy as np
import glob

def rotate_audio(audio, sr, n_beats):
    tempo, _ = beat.beat_track(audio, sr=sr, start_bpm=110, units='time', trim=False)
    samples_rotation = tempo * sr
    n_rotations = int(samples_rotation * n_beats)
    return np.roll(audio, n_rotations)


def adjust_tempo(song, final_tempo):
    #TODO: This function
    return song


def mix_songs(main_song, cand_song, beat_offset, pitch_shift):
    sr = 44100
    main_song, _ = core.load(main_song, sr=sr)
    cand_song, _ = core.load(cand_song, sr=sr)
    #Make everything mono
    if main_song.ndim == 2:
        main_song = np.sum(main_song, axis=1)/2
    if cand_song.ndim == 2:
        cand_song = np.sum(cand_song, axis=1)/2

    final_tempo, _ = beat.beat_track(main_song, start_bpm=110, units='time', trim=False)
    final_len = len(main_song)

    cand_song = adjust_tempo(cand_song, final_tempo)
    cand_song = effects.pitch_shift(cand_song, sr, -pitch_shift)
    beat_sr = final_tempo/(60 * sr)  # Number of samples per beat
    cand_song = cand_song[int(beat_offset*beat_sr):int(beat_offset*beat_sr + final_len)]
    return cand_song*0.5 + main_song*0.5