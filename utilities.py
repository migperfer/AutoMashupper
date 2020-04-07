from librosa import effects, core, output
import numpy as np
import essentia.standard as estd
from pyrubberband import pyrb


def match_target_amplitude(sound, target_dBFS=0):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def zapata14bpm(y):
    essentia_beat = estd.BeatTrackerMultiFeature()
    mean_tick_distance = np.mean(np.diff(essentia_beat(y)[0]))
    return 60/mean_tick_distance


def self_tempo_estimation(y, sr, tempo=None):
    """
    A function to calculate tempo based on a confidence measure
    :param y: The audio signal to which calculate the tempo
    :param sr: The sample rate of the signal
    :param tempo: Precalculated bpm
    :return: An array containing tempo, and an array of beats (in seconds)
    """
    if tempo is None:
        confidence_estimator = estd.LoopBpmConfidence(sampleRate=sr)
        percivalbpm = int(estd.PercivalBpmEstimator(sampleRate=sr)(y))
        try:
            zapatabpm = int(zapata14bpm(y))
        except:
            tempo = percivalbpm
        else:
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
    """
    Apply rotation to the audio in a given number of beats
    :param audio: The audio signal to rotate
    :param sr: The sample rate
    :param n_beats: Number of beats to rotate the audio
    :return:
    """
    tempo, _ = self_tempo_estimation(y, sr)
    samples_rotation = tempo * sr
    n_rotations = int(samples_rotation * n_beats)
    return np.roll(audio, n_rotations)


def adjust_tempo(song, final_tempo):
    """
    Adjust audio to the desired tempo
    :param song: The song which tempo should be adjusted
    :param final_tempo:
    :return:
    """
    actual_tempo, _ = self_tempo_estimation(song, 44100)
    song = pyrb.change_tempo(song, 44100, actual_tempo, final_tempo)
    """
    stretch_factor = final_tempo/actual_tempo
    if stretch_factor != 1:
    song = stretch(song, stretch_factor)
    """
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
    """
    Mixes two loops with a given beat_offset and a pitch_shift (applied to the candidate song)
    :param main_song: The path to the main loop
    :param cand_song: The path to the candidate loop
    :param beat_offset: The beat offset
    :param pitch_shift: The pitch shift
    :return: The resulting signal of the audio mixing with sr=44100
    """
    sr = 44100
    main_song, _ = core.load(main_song, sr=sr, mono=True)
    cand_song, _ = core.load(cand_song, sr=sr, mono=True)
    #Make everything mono
    final_tempo, _ = self_tempo_estimation(main_song, sr)
    final_len = len(main_song)

    beat_sr = final_tempo/(60 * sr)  # Number of samples per beat
    cand_song = cand_song[int(beat_offset*beat_sr):int(beat_offset*beat_sr + final_len)]
    # cand_song = effects.pitch_shift(cand_song, sr, -pitch_shift)
    tunning = np.mean(estd.TuningFrequencyExtractor()(cand_song))
    tunning_main = np.mean(estd.TuningFrequencyExtractor()(main_song))
    cand_song = adjust_tempo(cand_song, final_tempo)
    factor_tuning = tunning/tunning_main
    pitch_factor = factor_tuning*np.exp2(-pitch_shift/12)
    cand_song = pyrb.frequency_multiply(cand_song, 44100, pitch_factor)
    cand_song = core.resample(cand_song, 44100, 44100 / cand_song.shape[0] * main_song.shape[0])
    try:
        aux = np.zeros(main_song.shape[0])
        aux[:cand_song.shape[0]] = cand_song
        cand_song = aux
    except ValueError:
        aux = np.zeros(cand_song.shape[0])
        aux[:main_song.shape[0]] = main_song
        main_song = aux
    cand_song = cand_song.astype('float32')
    # main_song_replaygain = estd.ReplayGain()(main_song)
    # cand_song = estd.EqloudLoader(replayGain=main_song_replaygain)(cand_song)
    cand_song = cand_song/max(cand_song)
    main_song = main_song/max(main_song)
    return cand_song*0.5 + main_song*0.5

