import os
import shutil
import subprocess
import mir_utils.utils as utils
import argparse
import pretty_midi

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="source directory")
parser.add_argument("--align_dir", default='/home/ilcobo2/AlignmentTool_v2', help="source directory")
args = parser.parse_args()

os.chdir(args.align_dir)

audio_files = utils.find_files_in_subdirs(args.input_dir, '*.mp3')
audio_files.extend(utils.find_files_in_subdirs(args.input_dir, '*.wav'))

for audio_file in audio_files:
    if os.path.isfile(audio_file.replace('.mp3', '_infer_corresp.txt')):
        continue

    if audio_file in ["/dataset/jdl/Chopin Etude op. 10/11/Richter, Sviatoslav (2).mp3",
                      '/dataset/jdl/Chopin Etude op. 10/11/Cortot, Alfred.mp3',
                      '/dataset/jdl/Chopin Etude op. 10/11/Richter, Sviatoslav.mp3',
                      '/dataset/jdl/Chopin Etude op. 25/8/Lugansky, Nikolai.mp3',
                      '/dataset/jdl/Rachmaninov Etudes Tableaux/39-3/Richter, Sviatoslav.mp3',
                      '/dataset/jdl/Prokofiev Toccata op. 11/Cherkassky, Shura.mp3']:
        continue


    file_folder, file_name = utils.split_path_from_path(audio_file)
    infer_midi = audio_file.replace('.mp3', '_infer.mid')
    score_midi = os.path.join(file_folder, 'midi0.mid')

    mid = pretty_midi.PrettyMIDI(score_midi)

    n_notes = len(mid.instruments[0].notes)

    if n_notes >= 5000:
        continue
    print audio_file

    shutil.copy(infer_midi, os.path.join(args.align_dir, 'infer.mid'))
    shutil.copy(score_midi, os.path.join(args.align_dir, 'score.mid'))

    try:
        subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
    except:
        print 'Error to process {}'.format(audio_file)
        pass
    else:
        shutil.move('infer_corresp.txt', audio_file.replace('.mp3', '_infer_corresp.txt'))
        shutil.move('infer_match.txt', audio_file.replace('.mp3', '_infer_match.txt'))
        shutil.move('infer_spr.txt', audio_file.replace('.mp3', '_infer_spr.txt'))
        shutil.move('score_spr.txt', os.path.join(args.align_dir, '_score_spr.txt'))


