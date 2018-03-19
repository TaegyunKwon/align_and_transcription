from __future__ import division

import tgk_align_utils
import mir_utils.utils as utils
import os
import numpy as np



MISSING_RATIO_TH = 0.8
FPS = 16000/512
MIDI_MIN = 21


def time_to_frame(time, fps=FPS):
    return int(round(time * fps))


def note_to_frame(note, fps=16000 / 512):
    pitch = note.pitch
    time = note.time
    frame = time_to_frame(time)
    return (frame, pitch - MIDI_MIN)

def fill_in_match_onset(pseudo_onset, pseudo_onset_weight, note):
    frame_idx = note_to_frame(note)


def make_pseudo_label(frame_pred, onset_pred, corresp_notes):
    # try interpolation. if not, make uncertainty area




    ps_frame_label = np.zeros_like(frame)
    ps_onset_label = np.zeros_like(frame)
    ps_frame_weight = np.ones_like(frame)
    ps_onset_weight = np.ones_like(frame)

    # fill in matched notes




    return ps_frame_label, ps_onset_label, ps_frame_weight, ps_onset_weight


INPUT_DIR = '/dataset/jdl'

if __name__ == '__main__':

    files = utils.find_files_in_subdirs(INPUT_DIR, '*.mp3')

    for mp3_file in files:
        print mp3_file
        file_folder, file_name = utils.split_path_from_path(mp3_file)

        score_midi = os.path.join(file_folder, 'midi0.mid')
        frame = np.load(mp3_file.replace('.mp3', '_frame.npy'))
        onset = np.load(mp3_file.replace('.mp3', '_onset.npy'))
        corresp_file = mp3_file.replace('.mp3', '_infer_corresp.txt')
        if not os.path.isfile(corresp_file):
            continue
        notes = tgk_align_utils.read_corresp_result(corresp_file)
        n_match, n_miss, n_extra = tgk_align_utils.count_corresp_notes(notes)

        n_notes_score = n_match + n_miss

        if n_match / n_notes_score < MISSING_RATIO_TH:
            continue



