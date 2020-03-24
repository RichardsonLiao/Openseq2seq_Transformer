import librosa

import numpy as np
import tensorflow as tf
import copy

from open_seq2seq.utils.utils import  get_base_config, check_logdir, get_interactive_infer_results

args_S2T = ["--config_file=config_file/transformer_infer.py", "--mode=interactive_infer"]
#args_S2T = ["--config_file=/home/richardsonliao/inference/OpenSeq2Seq/config_file/w2lplus_thai.py", "--mode=interactive_infer"]
#args_S2T = ["--config_file=/home/richardsonliao/inference/OpenSeq2Seq/config_file/jasper10x5_LibriSpeech_nvgrad.py", "--mode=interactive_infer"]

def nested_update(org_dict, upd_dict):
  for key, value in upd_dict.items():
    if isinstance(value, dict):
      if key in org_dict:
        if not isinstance(org_dict[key], dict):
          raise ValueError(
              "Mismatch between org_dict and upd_dict at node {}".format(key)
          )
        nested_update(org_dict[key], value)
      else:
        org_dict[key] = value
    else:
      org_dict[key] = value

def get_model(args):
    args, base_config, base_model, config_module = get_base_config(args)
    checkpoint = check_logdir(args, base_config)
    # infer_config = copy.deepcopy(base_config)
    if args.mode == "interactive_infer":
        nested_update(base_config, copy.deepcopy(config_module['interactive_infer_params']))
    
    model = base_model(params=base_config, mode=args.mode, hvd=None)
    model.compile()
    return model, checkpoint, base_config["data_layer_params"]
# ================== start

# =================== end

model_S2T, checkpoint_S2T, data_layer_params = get_model(args_S2T)
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

# [v for v in tf.global_variables()]
# [n for n in tf.get_default_graph().get_operations()]
    

saver = tf.train.Saver()
saver.restore(sess, checkpoint_S2T)
saver.save(sess, '/home/richardsonliao/inference/STTInfer/STTInfer_data/NNModel/TF/tl/transformer/infer.ckpt')
#saver.save(sess, '/home/richardsonliao/inference/STTInfer/STTInfer_data/NNModel/TF/tl/jasper_112/infer.ckpt')
temp = get_interactive_infer_results(model_S2T, sess, ['/home/richardsonliao/inference/STTInfer/STTInfer_src/without the dataset the article is useless.wav'])[0][0]

for n in tf.get_default_graph().get_operations():
  print(n.name)
input(temp)


'''
Tensor("ForwardPass/fully_connected_ctc_decoder/logits:0", shape=(1, ?, 29), dtype=float16, device=/device:GPU:0)
Tensor("ForwardPass/fully_connected_ctc_decoder/transpose:0", shape=(?, 1, 29), dtype=float16, device=/device:GPU:0)
input_dict['encoder_output']['src_length']
Tensor("ForwardPass/ds2_encoder/floordiv_1:0", shape=(1,), dtype=int32, device=/device:GPU:0)
'''
'''
inputData = model_S2T.get_data_layer().input_tensors
print(inputData)
{'source_tensors': [
<tf.Tensor 'Placeholder:0' shape=(1, ?, 160) dtype=float16>,
<tf.Tensor 'Placeholder_1:0' shape=(1,) dtype=int32>],
'source_ids': [<tf.Tensor 'Placeholder_2:0' shape=(1,) dtype=int32>]
}
'''
'''
import pandas as pd
from os import listdir
mypath = '/media/sdb1/patricia_wav/audio'
files = listdir(mypath)
preds = []
filesName = []
for f in files:
  print(f)
  try:
    temp = get_interactive_infer_results(model_S2T, sess, [mypath + '/' + f])[0][0]
    preds.append(temp)
    filesName.append(f)
  except:
    preds.append('err')
    filesName.append(f)
pd.DataFrame(
    {
        'wav_filename': filesName,
        'predicted_transcript': preds,
    },
    columns=['wav_filename', 'predicted_transcript']
).to_csv('mytest.csv', index=False)
'''


'''
logits = tf.get_default_graph().get_tensor_by_name("ForwardPass/fully_connected_ctc_decoder/transpose:0")
# src_length = tf.get_default_graph().get_tensor_by_name("ForwardPass/ds2_encoder/floordiv_1:0")
model_in = ['/home/billy/catherine/audio/01_1.wav']
feed_dict = model_S2T.get_data_layer().create_feed_dict(model_in)
print(model_S2T.get_data_layer().params[idx2char])
# for 
softmax = tf.nn.softmax(logits, axis=-1)
softmax = tf.cast(softmax, tf.float32)
greedy = tf.nn.ctc_greedy_decoder(softmax, src_length)
greedy_ = sess.run(greedy, feed_dict=feed_dict)
print(greedy_[0][0].values)
# logits_, src_length_ = sess.run([logits, src_length], feed_dict=feed_dict)
# softmax_, src_length_ = sess.run([softmax, src_length], feed_dict=feed_dict)
# print(softmax_)
# print(src_length_)
'''
