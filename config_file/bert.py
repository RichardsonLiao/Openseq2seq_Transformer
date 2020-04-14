# pylint: skip-file
from __future__ import absolute_import, division, print_function
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TransformerEncoder
from open_seq2seq.decoders import TransformerDecoder
from open_seq2seq.decoders import JointCTCTransformerDecoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import MultiTaskCTCEntropyLoss, PaddedCrossEntropyLossWithSmoothing, CTCLoss
from open_seq2seq.data.text2text.text2text import SpecialTextTokens
from open_seq2seq.data.text2text.tokenizer import EOS_ID
from open_seq2seq.optimizers.lr_policies import poly_decay, transformer_policy
from open_seq2seq.optimizers.novograd import NovoGrad
import tensorflow as tf


"""
This configuration file describes a variant of Transformer model from
https://arxiv.org/abs/1706.03762
"""

base_model = Speech2Text
d_model = 16
num_layers = 10
batch_size = 10
vocab_size = 28

norm_params = {
  "type": "layernorm_L2",
  "momentum": 0.95,
  "epsilon": 0.00001,
}

#regularizer=tf.contrib.layers.l2_regularizer
#regularizer_params={'scale':0.001}

#norm_params = {
  #"type": "batch_norm",
  #"momentum": 0.95,
  #"epsilon": 0.00001,
  #"regularizer":regularizer,
  #"regularizer_params": regularizer_params,
#}

attention_dropout = 0.1
dropout = 0.3

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
#data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"
#data_root = "/data/wmt16-ende-sp/"

base_params = {
  "use_horovod": False,
  "num_gpus": 1, #8, # when using Horovod we set number of workers with params to mpirun
  "batch_size_per_gpu": batch_size,  # this size is in sentence pairs, reduce it if you get OOM
  "num_epochs":  1000,
  "save_summaries_steps": 10,
  "print_loss_steps": 10,
  "print_samples_steps": 10,
  "eval_steps": 10,
  "save_checkpoint_steps": 100,
  "logdir": "transformer-test",
  #"dtype": tf.float32, # to enable mixed precision, comment this line and uncomment two below lines
  "dtype": "mixed",
  "loss_scaling": "Backoff",

  "optimizer": NovoGrad,
  "optimizer_params": {
    "beta1": 0.95,
    "beta2": 0.99,
    "epsilon":  1e-08,
    "weight_decay": 0.00001,
    "grad_averaging": False,
  },
  #"lr_policy": transformer_policy,
  #"lr_policy_params": {
    #"learning_rate": 2.0,
    #"warmup_steps": 200,
    #"d_model": d_model,
  #},
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "power": 2.0,
    "min_lr": 1e-5,
  },

  "larc_params": {
    "larc_eta": 0.001,
  },

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
              'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": TransformerEncoder,
  "encoder_params": {
    "batch_size": batch_size,
    "num_features": 64,
    "src_vocab_size": vocab_size,
    "encoder_layers": num_layers,
    "hidden_size": d_model,
    "num_heads": 16,
    "filter_size": 4 * d_model,
    "attention_dropout": attention_dropout,  # 0.1,
    "relu_dropout": dropout,                 # 0.3,
    "layer_postprocess_dropout": dropout,    # 0.3,
    "pad_embeddings_2_eight": False,
    "remove_padding": True,
    "norm_params": norm_params,
  },

  "decoder": FullyConnectedCTCDecoder,
  "decoder_params": {
    "initializer": tf.contrib.layers.xavier_initializer,
    "use_language_model": False,
    "tgt_vocab_size": vocab_size,
    #"regularizer": tf.contrib.layers.l2_regularizer,
    #"regularizer_params": {'scale':0.001}
  },

  "loss": CTCLoss,
  "loss_params": {
  },
  #"loss": MultiTaskCTCEntropyLoss,
  #"loss_params": {
  #  "seq_loss_params": {
  #    "offset_target_by_one": False,
  #    "average_across_timestep": True,
  #    "do_mask": False
  #  },

  #  "ctc_loss_params": {
  #  },

  #  "lambda_value": 0.25,
  #  "tgt_vocab_size": vocab_size,
  #  "batch_size": batch_size,
  #},

  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "num_audio_features": 64,
    "input_type": "logfbank",
    "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
    "norm_per_feature": True,
    "window": "hanning",
    "precompute_mel_basis": True,
    "sample_freq": 16000,
    "pad_to": 16,
    "dither": 1e-5,
    "backend": "librosa"
  },
}

train_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "dataset_files": [
      "open_seq2seq/test_utils/toy_speech_data/toy_data.csv"
      #"open_seq2seq/test_utils/toy_speech_data/one-sentence.csv",
    ],
    "max_duration": 16.7,
    "shuffle": False,
  },
}

eval_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "dataset_files": [
      "open_seq2seq/test_utils/toy_speech_data/toy_data.csv"
      #"open_seq2seq/test_utils/toy_speech_data/one-sentence.csv",
    ],
    "shuffle": False,
    },
}

infer_params = {
  "data_layer": Speech2TextDataLayer,
  "data_layer_params": {
    "dataset_files": [
      "open_seq2seq/test_utils/toy_speech_data/toy_data.csv"
      #"open_seq2seq/test_utils/toy_speech_data/one-sentence.csv",
    ],
    "shuffle": False,
  },
}

