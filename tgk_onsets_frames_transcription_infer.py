# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference for onset conditioned model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import re
import time

# internal imports

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim as slim

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
import infer_util
from magenta.models.onsets_frames_transcription import model
from magenta.music import midi_io

import mir_utils.utils as utils
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'acoustic_run_dir', None,
    'Path to look for acoustic checkpoints. Should contain subdir `train`.')
tf.app.flags.DEFINE_string(
    'acoustic_checkpoint_filename', None,
    'Filename of the checkpoint to use. If not specified, will use the latest '
    'checkpoint')
tf.app.flags.DEFINE_string(
    'examples_path', None,
    'Path to TFRecord of test examples.')
tf.app.flags.DEFINE_string(
    'run_dir', '~/tmp/onsets_frames/infer',
    'Path to store output midi files and summary events.')
tf.app.flags.DEFINE_string(
    'hparams',
    'onset_mode=length_ms,onset_length=32',
    'A comma-separated list of `name=value` hyperparameter values.')
tf.app.flags.DEFINE_float(
    'note_threshold', 0.5,
    'Threshold to use when sampling from the acoustic model.')
tf.app.flags.DEFINE_integer(
    'max_seconds_per_sequence', 0,
    'If set, will truncate sequences to be at most this many seconds long.')
tf.app.flags.DEFINE_integer(
    'min_note_duration_ms', 0,
    'Notes shorter than this duration will be ignored when computing metrics.')
tf.app.flags.DEFINE_boolean(
    'require_onset', True,
    'If set, require an onset prediction for a new note to start.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def model_inference(acoustic_checkpoint, hparams, examples_path, run_dir):
  tf.logging.info('acoustic_checkpoint=%s', acoustic_checkpoint)
  tf.logging.info('examples_path=%s', examples_path)
  tf.logging.info('run_dir=%s', run_dir)


  with tf.Graph().as_default():
    num_dims = constants.MIDI_PITCHES

    # Build the acoustic model within an 'acoustic' scope to isolate its
    # variables from the other models.
    with tf.variable_scope('acoustic'):
      truncated_length = 0
      if FLAGS.max_seconds_per_sequence:
        truncated_length = int(
            math.ceil((FLAGS.max_seconds_per_sequence *
                       data.hparams_frames_per_second(hparams))))
      acoustic_data_provider = data.provide_batch(
          batch_size=1,
          examples_path=examples_path,
          hparams=hparams,
          is_training=False,
          truncated_length=truncated_length)

      _, _, _, acoustic_logits, _ = model.get_model(
          acoustic_data_provider, hparams, is_training=False)

    acoustic_variables = {
      re.sub(r'^acoustic/', '', var.op.name): var
      for var in slim.get_variables(scope='acoustic/')
    }
    acoustic_restore = tf.train.Saver(acoustic_variables)

    onset_probs_flat = tf.get_default_graph().get_tensor_by_name(
        'acoustic/onsets/onset_probs_flat:0')
    def init_fn(unused_self, sess):
      acoustic_restore.restore(sess, acoustic_checkpoint)

    scaffold = tf.train.Scaffold(init_fn=init_fn)
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold)
    sess = tf.train.MonitoredSession(session_creator=session_creator)
    # Use a custom init function to restore the acoustic and language models
    # from their separate checkpoints.
    tf.logging.info('Inferring for %d batches',
                  acoustic_data_provider.num_batches)
    infer_times = []
    num_frames = []
    for unused_i in range(acoustic_data_provider.num_batches):
        start_time = time.time()
        filenames, note_sequences, logits, onset_logits = sess.run([
            acoustic_data_provider.filenames,
            acoustic_data_provider.note_sequences,
            acoustic_logits,
            onset_probs_flat,
            ])
        # We expect these all to be length 1 because batch size is 1.
        assert len(filenames) == len(note_sequences) == 1

        end_time = time.time()
        infer_time = end_time - start_time
        infer_times.append(infer_time)
        num_frames.append(logits.shape[0])
        tf.logging.info(
            'Infer time %f, frames %d, frames/sec %f, running average %f',
            infer_time, logits.shape[0],
            logits.shape[0] / infer_time,
            np.sum(num_frames) / np.sum(infer_times))
        normal_name = filenames[0].replace('.mp3', '')
        np.save(normal_name + '_logits.npy', logits)
        np.save(normal_name +  '_onsets.npy', onset_logits)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.acoustic_checkpoint_filename:
    acoustic_checkpoint = os.path.join(
        os.path.expanduser(FLAGS.acoustic_run_dir), 'train',
        FLAGS.acoustic_checkpoint_filename)
  else:
    acoustic_checkpoint = tf.train.latest_checkpoint(
        os.path.join(os.path.expanduser(FLAGS.acoustic_run_dir), 'train'))

  run_dir = os.path.expanduser(FLAGS.run_dir)

  hparams = tf_utils.merge_hparams(
      constants.DEFAULT_HPARAMS, model.get_default_hparams())
  hparams.parse(FLAGS.hparams)

  tf.gfile.MakeDirs(run_dir)

  model_inference(
      acoustic_checkpoint=acoustic_checkpoint,
      hparams=hparams,
      examples_path=FLAGS.examples_path,
      run_dir=run_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
