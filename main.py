from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import random

import modeling
import optimization
import tokenization
import tensorflow as tf

from tensorflow.contrib import tpu
from estimator_utils import model_fn_builder, input_fn_builder, convert_ids_to_features, EvalHook

flags = tf.flags

FLAGS = flags.FLAGS


class Config:
    data_dir = './data'
    bert_dir = './bert_data/'
    bert_config_file = bert_dir + 'bert_config.json'
    task_name = 'MNLI'
    vocab_file = bert_dir + 'vocab.txt'
    output_dir = './output_model'
    init_checkpoint = bert_dir + 'bert_model.ckpt'
    do_lower_case = True
    max_seq_length = 512
    margin = 5.0
    do_train = True
    do_eval = False
    train_batch_size = 2
    eval_batch_size = 2
    learning_rate = 2e-5
    num_train_epochs = 5.0
    warmup_proportion = 0.1
    save_checkpoints_steps = 100
    iterations_per_loop = 100


config = Config()

flags.DEFINE_string("data_dir", config.data_dir,
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("bert_config_file", config.bert_config_file,
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")
flags.DEFINE_string("vocab_file", config.vocab_file,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", config.output_dir,
                    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string("init_checkpoint", config.init_checkpoint,
                    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", config.do_lower_case,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")
flags.DEFINE_integer("max_seq_length", config.max_seq_length,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded.")
flags.DEFINE_float("margin", config.margin,
                   "margin.")

flags.DEFINE_bool("do_train", config.do_train, "Whether to run training.")
flags.DEFINE_bool("do_eval", config.do_eval, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", config.train_batch_size, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", config.eval_batch_size, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", config.learning_rate, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", config.num_train_epochs, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", config.warmup_proportion,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", config.save_checkpoints_steps,
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", config.iterations_per_loop,
                     "How many steps to make in each estimator call.")


def train(bert_config, run_config):
    train_file = os.path.join(FLAGS.data_dir, "data.json")
    if os.path.exists(train_file) is False:
        raise ValueError("Could find training file!")
    with open(train_file) as fr:
        data = json.load(fr)
        train_ids = data["train_ids"]
        dev_ids = data["dev_ids"]

    num_train_steps = int(len(train_ids) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    rng = random.Random(12345)
    rng.shuffle(train_ids)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        margin=FLAGS.margin)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.eval_batch_size)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_features = convert_ids_to_features(token_ids=train_ids, tokenizer=tokenizer,
                                             max_seq_length=FLAGS.max_seq_length, is_training=True)
    dev_features = convert_ids_to_features(token_ids=dev_ids, tokenizer=tokenizer,
                                           max_seq_length=FLAGS.max_seq_length, is_training=True)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num train examples = %d", len(train_ids))
    tf.logging.info("  Num dev examples = %d", len(dev_ids))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = input_fn_builder(input_features=train_features, seq_length=FLAGS.max_seq_length,
                                      is_training=True, drop_remainder=True)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps,
                    hooks=[EvalHook(estimator=estimator,
                                    eval_features=dev_features,
                                    max_seq_length=FLAGS.max_seq_length,
                                    eval_steps=FLAGS.save_checkpoints_steps,
                                    save_model_dir="save_model",
                                    th=85.0,
                                    output_dir=FLAGS.output_dir)])


def predict(bert_config, run_config):
    pass


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    tf.gfile.MakeDirs(FLAGS.output_dir)

    is_per_host = tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tpu.RunConfig(tpu_config=tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                                                        num_shards=8,
                                                        per_host_input_for_training=is_per_host),
                               model_dir=FLAGS.output_dir,
                               save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config = run_config.replace(session_config=session_config)
    run_config = run_config.replace(keep_checkpoint_max=2)

    if FLAGS.do_train:
        train(bert_config, run_config)

    if FLAGS.do_eval:
        predict(bert_config, run_config)


if __name__ == "__main__":
    main()
