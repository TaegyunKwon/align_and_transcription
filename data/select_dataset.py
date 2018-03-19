from __future__ import division

import os
import random
import shutil
from mir_utils import utils


DATA_SIZE = 600
DATA_PATH = '/dataset/sourceFiles'
MAX_PICK_PER_PIECE = 5

def get_paths(root):
    midi_lists = list()
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith('.mid'):
                full_path = os.path.join(path, name)
                midi_lists.append(full_path)
    return midi_lists


if __name__ == '__main__':
    candidates_folders = list()
    for path, subdirs, files in os.walk(DATA_PATH):
        for name in files:
            if name == '(midi)_clean.mid':
                candidates_folders.append(path)

    random.shuffle(candidates_folders)

    candidate_lists = list()
    n_candidates = 0
    for folder in candidates_folders:
        if n_candidates >= DATA_SIZE:
            break
        files = list()
        for name in os.listdir(folder):
            if name.endswith('.mp3') and 'midi' not in name and '(Cembalo)' not in name:
                files.append(name)
        random.shuffle(files)
        n_max = min(len(files), MAX_PICK_PER_PIECE)
        n_candidates += n_max
        for n in range(n_max):
            candidate_lists.append(os.path.join(folder, files[n]))

    n_valid = n_test = int(1/5 * DATA_SIZE)
    n_train = len(candidate_lists) - n_valid - n_test

    def write_list(name, candidates):
        f = open(name, 'wb')
        for el in candidates:
            f.write(el + '\n')
        f.close()

    write_list('jdl_600.txt', candidate_lists)

    for el in candidate_lists:
        file_with_subdir = el.replace('/dataset/sourceFiles/', '')
        subdirs, filename = utils.split_path_from_path(file_with_subdir)
        new_path = '/dataset/jdl/' + file_with_subdir
        utils.maybe_make_dir(os.path.join('/dataset/jdl/', subdirs))
        shutil.copy(el, new_path)
        if not os.path.isfile(os.path.join('/dataset/jdl/', subdirs, '(midi).mid')):
            shutil.copy(os.path.join('/dataset/sourceFiles/', subdirs, '(midi).mid'),
                        os.path.join('/dataset/jdl/', subdirs, '(midi).mid'))




