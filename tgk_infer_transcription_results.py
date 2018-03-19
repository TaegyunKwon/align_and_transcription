from __future__ import division

import tgk_align_utils
from mir_utils import utils
import argparse
import re
import numpy as np
import infer_util
from magenta.music import midi_io


def find_and_concat_transcription_result(audio_file, save=False):
    def get_seg_num(file_name):
        m = re.search('seg(.+?).npy', file_name)
        if m:
            found = m.group(1)
        return int(found)

    def sort_segments(file_list):
        seg_num = [get_seg_num(el) for el in file_list]
        return [el for _, el in sorted(zip(seg_num, file_list))]

    file_folder, file_name = utils.split_path_from_path(audio_file)
    frame_files = utils.find_files_in_subdirs(
        file_folder, file_name.replace('.mp3', '_logits.npy*'))
    onset_files = utils.find_files_in_subdirs(
        file_folder, file_name.replace('.mp3', '_onsets.npy*'))

    frame_files = sort_segments(frame_files)
    onset_files = sort_segments(onset_files)

    frames = [np.load(el) for el in frame_files]
    onsets = [np.load(el) for el in onset_files]

    frame_concat = tgk_align_utils.concat_overlap_logits(frames)
    onset_concat = tgk_align_utils.concat_overlap_logits(onsets)
    if save:
        np.save(audio_file.replace('.mp3', '_frame.npy'), frame_concat)
        np.save(audio_file.replace('.mp3', '_onset.npy'), onset_concat)
    return frame_concat, onset_concat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="source directory")
    parser.add_argument("--save_dir", help="save directory")
    args = parser.parse_args()

    audio_files = utils.find_files_in_subdirs(args.input_dir, '*.mp3')
    audio_files.extend(utils.find_files_in_subdirs(args.input_dir, '*.wav'))

    for audio_file in audio_files:
        print audio_file
        frame_infer, onset_infer = find_and_concat_transcription_result(audio_file, save=True)
        frame_predictions = frame_infer > 0.5
        onset_predictions = onset_infer > 0.5

        sequence_prediction = infer_util.pianoroll_to_note_sequence(
            frame_predictions,
            frames_per_second=16000/512,
            min_duration_ms=0,
            onset_predictions=onset_predictions)
        output_file = audio_file.replace('.mp3', '_infer.mid')
        midi_io.sequence_proto_to_midi_file(sequence_prediction, output_file)


