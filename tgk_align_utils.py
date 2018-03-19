from __future__ import division

import numpy as np
import pretty_midi
from fastdtw import fastdtw
import mir_utils.midi_utils as midi_utils
import mir_utils.utils as utils
import mido


def concat_overlap_logits(logit_list, frame_len=3750, overlap_len=625):
    non_overlap_len = frame_len - overlap_len
    total_len = (len(logit_list) - 1) * non_overlap_len + logit_list[-1].shape[0]
    concat = np.zeros((total_len, logit_list[0].shape[1]))

    # fill in first segment
    concat[:non_overlap_len, :] = logit_list[0][:non_overlap_len, :]
    for n in range(1, len(logit_list)):
        #  simple fill in (non overlap)
        concat[n * non_overlap_len: n * non_overlap_len + overlap_len//2, :] = \
            logit_list[n-1][non_overlap_len: non_overlap_len + overlap_len//2, :]
        concat[n * non_overlap_len + overlap_len//2: n * non_overlap_len + overlap_len, :] = \
            logit_list[n][overlap_len//2: overlap_len, :]
        if n != len(logit_list) - 1:
            # fill in middle part
            concat[n * non_overlap_len + overlap_len: (n + 1) * non_overlap_len, :] = \
                logit_list[n][overlap_len: non_overlap_len, :]
        else:
            concat[n * non_overlap_len + overlap_len:, :] = \
                logit_list[-1][overlap_len:, :]

    return concat


class MatchedNote():
    def __init__(self):
        self.id = None
        self.onset_time = None
        self.pitch = None

        self.status = None
        self.score_time = None
        self.score_id = None
        self.score_pitch = None

    def __repr__(self):
        return 'id: {}, start:{:0.4f}, pitch:{:d}, status:{}'.\
            format(self.id, self.onset_time, self.pitch, self.status)

    def validate(self):
        return self.status in ['match', 'extra', 'missing']


def read_corresp_result(corresp_file):
    f = open(corresp_file, 'rb')
    lines = f.readlines()[1:]
    notes = []

    for line in lines:
        note = MatchedNote()
        line_split = line.split()

        note.id = line_split[0]
        note.onset_time = float(line_split[1])
        note.pitch = int(line_split[3])

        note.score_id = line_split[5]
        note.score_time = float(line_split[6])
        note.score_pitch = int(line_split[8])

        if note.id == '*':
            note.status = 'missing'
        elif note.score_id == '*' or note.pitch != note.score_pitch:
            note.status = 'extra'
        else:
            note.status = 'match'

        notes.append(note)
    return notes


def remove_extra_notes(infer_midi, corresp_file, save=False):
    match_notes = read_corresp_result(corresp_file)

    mid = pretty_midi.PrettyMIDI(infer_midi)
    midi_notes = mid.instruments[0].notes

    delete_notes = []
    for n in range(len(midi_notes)):
        midi_note = midi_notes[n]
        note_time = midi_note.start
        note_pitch = midi_note.pitch
        for match_note in match_notes:
            if abs(float(match_note.onset_time) - note_time) <= 1e-3 and note_pitch == match_note.pitch:
                if match_note.status == 'extra':
                    delete_notes.append(n)
                break
            else:
                pass
    for index in sorted(delete_notes, reverse=True):
        del midi_notes[index]
    if save:
        mid.write(infer_midi.replace('.mid', '_rm.mid'))
    return mid


def count_corresp_notes(notes):
    n_match = n_miss = n_extra = 0

    for note in notes:
        if note.status == 'match':
            n_match += 1
        elif note.status == 'extra':
            n_extra += 1
        elif note.status == 'missing':
            n_miss += 1
    return n_match, n_miss, n_extra


def forced_align(score_midi, onset_pred, frame_pred, save=False, save_name=None):
    def midi_to_template(midi_file):
        midi_frame = midi_utils.mid2piano_roll(midi_file, fps=16000/512)
        midi_onset = midi_utils.piano_roll2chroma_roll(midi_utils.mid2piano_roll(midi_file, onset=True, fps=16000/512))
        midi_onset = utils.onset2delayed(midi_onset, delay_len=5)

        return np.concatenate([midi_frame, midi_onset], axis=1)

    def pred_to_template(onset_pred, frame_pred):
        onset_pred = midi_utils.piano_roll2chroma_roll(onset_pred)
        onset_pred = utils.onset2delayed(onset_pred, delay_len=5)

        return np.concatenate([frame_pred, onset_pred], axis=1)

    def euclidean_with_onset_ro(y1, y2, onset_weight=8):
        dist1 = np.linalg.norm(y1[:88] - y2[:88])
        dist2 = np.linalg.norm(y1[88:] - y2[88:])

        return dist1 + onset_weight * dist2

    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(np.sum(targets * np.log(predictions + 1e-9))) / N
        return ce

    def depad(path, maxlen):
        path -= 100
        path[path < 0] = 0
        path[path > maxlen] = maxlen
        return path

    def align_midi_from_path(path_x, path_y, midi_file):
        def tick2time(tick, tempo, TPB):
            return tick / TPB * tempo / 1000000

        def time2tick(time, tempo, TPB):
            return int(round(time * TPB * 1000000 / tempo))
        mid = mido.MidiFile(midi_file)
        mid_align = mido.MidiFile(midi_file)

        t = 0
        t_align = 0
        TPB = mid.ticks_per_beat
        # tempo = mid.tracks[0][0].tempo
        tempo = 500000

        for k in xrange(len(mid.tracks[1])):
            message = mid.tracks[1][k]
            t_new = t + tick2time(message.time, tempo, TPB)
            if mid.tracks[1][k].time != 0:
                align_frame_x = int(t_new * 16000/512)
                arg_x = np.where(path_x == align_frame_x)[0][0]
                y_est = path_y[arg_x]
                t_est = y_est / (16000 / 512)
                t_diff = t_est - t_align
                tick_diff = time2tick(t_diff, tempo, TPB)
                mid_align.tracks[1][k].time = tick_diff
                t = t_new
                t_diff_quantized = tick2time(tick_diff, tempo, TPB)
                t_align += t_diff_quantized
        return mid_align

    midi_template = midi_to_template(score_midi)
    midi_template = np.pad(midi_template, ((100, 100), (0, 0)), 'constant', constant_values=0)
    pred_template = pred_to_template(onset_pred, frame_pred)
    pred_template = np.pad(pred_template, ((100, 100), (0, 0)), 'constant', constant_values=0)

    dist, path = fastdtw(midi_template, pred_template, dist=euclidean_with_onset_ro)
    # dist, path = fastdtw(midi_template, pred_template, dist=cross_entropy)
    path = np.asarray(path)
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_x = depad(path_x, midi_template.shape[0] - 100)
    path_y = depad(path_y, pred_template.shape[0] - 100)

    mid_align = align_midi_from_path(path_x, path_y, score_midi)

    if save:
        mid_align.save(save_name)
        np.savez(save_name.replace('.mid', '_align_path_gg.npz'), dist=dist, path=path)

    return mid_align, path, dist



