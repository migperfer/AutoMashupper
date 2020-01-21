from librosa import effects, core, output
import numpy as np
import essentia.standard as estd
import pyaudio


def playaudio(y, sr):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    frames_per_buffer=1024,
                    output=True,
                    output_device_index=0
                    )
    stream.write(y.tostring())
    stream.close()


def zapata14bpm(y):
    essentia_beat = estd.BeatTrackerMultiFeature()
    mean_tick_distance = np.mean(np.diff(essentia_beat(y)[0]))
    return 60/mean_tick_distance


def self_tempo_estimation(y, sr):
    confidence_estimator = estd.LoopBpmConfidence(sampleRate=sr)
    percivalbpm = int(estd.PercivalBpmEstimator(sampleRate=sr)(y))
    zapatabpm = int(zapata14bpm(y))
    confidence_zapata = confidence_estimator(y, zapatabpm)
    confidence_percival = confidence_estimator(y, percivalbpm)
    if confidence_percival >= confidence_zapata:
        tempo = percivalbpm
    else:
        tempo = zapatabpm

    sec_beat = (60/tempo)
    beats = np.arange(0, len(y)/sr, sec_beat)
    return tempo, beats


def rotate_audio(audio, sr, n_beats):
    tempo, _ = self_tempo_estimation(audio, sr)
    samples_rotation = tempo * sr
    n_rotations = int(samples_rotation * n_beats)
    return np.roll(audio, n_rotations)


def adjust_tempo(song, final_tempo):
    actual_tempo, _ = self_tempo_estimation(song, 44100)
    stretch_factor = final_tempo/actual_tempo
    if stretch_factor != 1:
        song = stretch(song, stretch_factor)
    return song


def stretch(x, factor, nfft=2048):
    '''
    From this repository: https://github.com/gaganbahga/time_stretch
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

    final_tempo, _ = self_tempo_estimation(main_song, sr)
    final_len = len(main_song)

    cand_song = adjust_tempo(cand_song, final_tempo)
    cand_song = effects.pitch_shift(cand_song, sr, -pitch_shift)
    beat_sr = final_tempo/(60 * sr)  # Number of samples per beat
    cand_song = cand_song[int(beat_offset*beat_sr):int(beat_offset*beat_sr + final_len)]
    cand_song = core.resample(cand_song, 44100, 44100 / cand_song.shape[0] * main_song.shape[0])
    try:
        aux = np.zeros(main_song.shape[0])
        aux[:cand_song.shape[0]] = cand_song
        cand_song = aux
    except ValueError:
        aux = np.zeros(cand_song.shape[0])
        aux[:main_song.shape[0]] = main_song
        main_song = aux

    return cand_song*0.5 + main_song*0.5

