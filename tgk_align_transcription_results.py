import tgk_align_utils
import mir_utils.utils as utils
import os
import numpy as np


INPUT_DIR = '/dataset/jdl'

if __name__ == '__main__':

    files = utils.find_files_in_subdirs(INPUT_DIR, '*.mp3')

    for el in files:
        print el
        file_folder, file_name = utils.split_path_from_path(el)

        score_midi = os.path.join(file_folder, 'midi0.mid')
        frame = np.load(el.replace('.mp3', '_frame.npy'))
        onset = np.load(el.replace('.mp3', '_onset.npy'))
        tgk_align_utils.forced_align(
            score_midi, onset, frame, save=True, save_name=el.replace('.mp3', '_align_eu.mid'))

