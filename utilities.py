from librosa import beat, effects, core, output
import numpy as np
import glob

def rotate_audio(audio, sr, n_beats):
    tempo, _ = beat.beat_track(audio, sr=sr, start_bpm=110, units='time', trim=False)
    samples_rotation = tempo * sr
    n_rotations = int(samples_rotation * n_beats)
    return np.roll(audio, n_rotations)


def adjust_tempo(song, final_tempo):
    actual_tempo, _ = beat.beat_track(song, start_bpm=110, units='time', trim=False)
    stretch_factor = final_tempo/actual_tempo
    song = stretch(song, stretch_factor)
    return song


def stretch(x, factor, nfft=2048):
    '''
    stretch an audio sequence by a factor using FFT of size nfft converting to frequency domain
    :param x: np.ndarray, audio array in PCM float32 format
    :param factor: float, stretching or shrinking factor, depending on if its > or < 1 respectively
    :return: np.ndarray, time stretched audio
    '''
    stft = core.stft(x, n_fft=nfft).transpose()  # i prefer time-major fashion, so transpose
    stft_rows = stft.shape[0]
    stft_cols = stft.shape[1]

    times = np.arange(0, stft.shape[0], factor)  # times at which new FFT to be calculated
    hop = nfft/4                                 # frame shift
    stft_new = np.zeros((len(times), stft_cols), dtype=np.complex_)
    phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ nfft
    phase = np.angle(stft[0])

    stft = np.concatenate( (stft, np.zeros((1, stft_cols))), axis=0)

    for i, time in enumerate(times):
        left_frame = int(np.floor(time))
        local_frames = stft[[left_frame, left_frame + 1], :]
        right_wt = time - np.floor(time)                        # weight on right frame out of 2
        local_mag = (1 - right_wt) * np.absolute(local_frames[0, :]) + right_wt * np.absolute(local_frames[1, :])
        local_dphi = np.angle(local_frames[1, :]) - np.angle(local_frames[0, :]) - phase_adv
        local_dphi = local_dphi - 2 * np.pi * np.floor(local_dphi/(2 * np.pi))
        stft_new[i, :] =  local_mag * np.exp(phase*1j)
        phase += local_dphi + phase_adv

    return core.istft(stft_new.transpose())


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