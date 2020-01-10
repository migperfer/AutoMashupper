import numpy as np
from scipy import signal
from segmentation import get_beat_sync_chroma, get_beat_sync_spectrums
import glob


def mashability(base_beat_sync_chroma, base_beat_sync_spec, audio_file_candidate):
    """
    Calculate mashability for a beat_sync_chroma and a audio_file_candidate
    returns the 2dconv_matrix normalised of the two_chromas, the max mashability achieved,
    the
    """
    # 1st step: Calculate harmonic compatibility
    c_bsc = np.flip(get_beat_sync_chroma(audio_file_candidate), )  # Flip to ensure max mashability with
    stacked_beat_sync_chroma = np.vstack([c_bsc, c_bsc])
    conv = signal.convolve2d(stacked_beat_sync_chroma, base_beat_sync_chroma, )
    base_n = np.linalg.norm(base_beat_sync_chroma)
    cand_n = np.linalg.norm(c_bsc)
    h_mas = conv / (base_n * cand_n)
    offset = base_beat_sync_chroma.shape[1]
    offset -= 1 if offset % 2 == 0 else 2
    h_mas_k = np.max(h_mas[:, offset:-offset], axis=0)  # Get only valid values on beat axis

    # 3rd step: Calculate Spectral balance compatibility
    c_bss = get_beat_sync_spectrums(audio_file_candidate)

    if c_bss.shape[1] >= base_beat_sync_spec.shape[1]:
        beat_length = base_beat_sync_spec.shape[1]
        n_max_b_shifts = c_bss.shape[1] - base_beat_sync_spec.shape[1]
        r_mas_k = np.zeros(n_max_b_shifts+1)
        for i in range(n_max_b_shifts+1):
            beta = np.sum(base_beat_sync_spec + c_bss[i:i + beat_length - 1], axis=0)
            beta_norm = beta/np.sum(beta)
            r_mas_k[i] = 1 - np.std(beta_norm)
    else:
        raise Exception("Candidate song has lesser beats than base song")


    res_mash = h_mas_k + 0.2 * r_mas_k

    b_offset = np.argmax(res_mash)
    p_shift = 7 - np.argmax(conv[11:-11, b_offset])
    return conv, np.max(res_mash), p_shift, b_offset

if __name__ == '__main__':
    base_song = "audio_files/looperman-l-0000003-0106600-serialchiller-sprung-spring-piano-arp.mp3"
    base_schroma = get_beat_sync_chroma(base_song)
    base_spec = get_beat_sync_spectrums(base_song)
    songs = glob.glob("audio_files/*.mp3")
    mashabilities = {}
    for cand_song in songs:
        try:
             mashabilities[cand_song] = mashability(base_schroma,
                                                base_spec,
                                                cand_song)[1:]
        except Exception as e:
            print("Skipping song %s, because %s" % (cand_song, str(e)))