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
from open_seq2seq.optimizers.lr_policies import poly_decay
from open_seq2seq.optimizers.novograd import NovoGrad
import tensorflow as tf


"""
This configuration file describes a variant of Transformer model from
https://arxiv.org/abs/1706.03762
"""

base_model = Speech2Text
d_model = 128
num_layers = 3
batch_size = 1
vocab_size = 29

norm_params = {
  "type": "layernorm_L2",
  "momentum": 0.95,
  "epsilon": 0.00001,
}

attention_dropout = 0.1
dropout = 0.3

# REPLACE THIS TO THE PATH WITH YOUR WMT DATA
#data_root = "[REPLACE THIS TO THE PATH WITH YOUR WMT DATA]"
#data_root = "/data/wmt16-ende-sp/"

base_params = {
  "use_horovod": False,
  "num_gpus": 1, #8, # when using Horovod we set number of workers with params to mpirun
  "batch_size_per_gpu": batch_size,  # this size is in sentence pairs, reduce it if you get OOM
  "max_steps":  10000,
  "save_summaries_steps": 1,
  "print_loss_steps": 1,
  "print_samples_steps": 10,
  "eval_steps": 100,
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
  "lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "power": 2,
  },

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

  "decoder": JointCTCTransformerDecoder,
  "decoder_params": {
    "attn_decoder": TransformerDecoder,
    "attn_decoder_params": {
      "batch_size": batch_size,
      "num_features": 64,
      "tgt_vocab_size": vocab_size,
      "num_hidden_layers": num_layers,
      "hidden_size": d_model,
      "num_heads": 16,
      "filter_size": 4 * d_model,
      "attention_dropout": attention_dropout,  # 0.1,
      "relu_dropout": dropout,  # 0.3,
      "layer_postprocess_dropout": dropout,  # 0.3,
      "beam_size": 4,
      "alpha": 0.6,
      "extra_decode_length": 50,
      "EOS_ID": EOS_ID,
      "norm_params": norm_params,
    },

    "ctc_decoder": FullyConnectedCTCDecoder,
    "ctc_decoder_params": {
      "initializer": tf.contrib.layers.xavier_initializer,
      "use_language_model": False,
      "tgt_vocab_size": vocab_size,
    },

    "beam_search_params": {
      #"beam_width": 4,
    },

    "language_model_params": {
      # params for decoding the sequence with language model
      #"use_language_model": False,
    },

  },

  "loss": MultiTaskCTCEntropyLoss,
  "loss_params": {
    "seq_loss_params": {
      "offset_target_by_one": False,
      "average_across_timestep": True,
      "do_mask": False
    },

    "ctc_loss_params": {
    },

    "lambda_value": 0.25,
    "tgt_vocab_size": vocab_size,
    "batch_size": batch_size,
  },

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

