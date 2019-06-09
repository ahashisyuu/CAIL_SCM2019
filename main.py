from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import random

from tqdm import tqdm

import modeling
import optimization
import tokenization
import tensorflow as tf

from tensorflow.contrib import tpu
from estimator_utils import model_fn_builder, input_fn_builder, \
    convert_ids_to_features, convert_ids_to_features_v2, EvalHook
from data_process.split_data import Example, convert_examples

flags = tf.flags

FLAGS = flags.FLAGS


class Config:
    data_dir = './data'
    data_file = "data.json"
    # test_file = "../input/input.txt"
    test_file = "./data/input.txt"
    # bert_dir = './model/bert/'
    bert_dir = "bert_data/"
    bert_config_file = bert_dir + 'bert_config.json'
    vocab_file = bert_dir + 'vocab.txt'
    output_dir = './output_model'
    init_checkpoint = bert_dir + 'bert_model.ckpt'
    # init_checkpoint = None
    do_lower_case = True
    max_seq_length = 511
    margin = 5.0
    do_pro = False
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
flags.DEFINE_string("data_file", config.data_file,
                    "data file")
flags.DEFINE_string("test_file", config.test_file,
                    "test file")
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

flags.DEFINE_bool("do_pro", config.do_pro, "Whether to run pro.")
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


def process():
    data_file = "./data/input.txt"

    all_examples = []

    with open(data_file, encoding='utf-8') as fr:
        count = 0
        for line in fr:
            data_line = json.loads(line)
            A = data_line["A"].split('\n')[2]
            B = data_line["B"].split('\n')[2]
            C = data_line["C"].split('\n')[2]
            count += 1

            example = Example(A=A, B=B, C=C)

            all_examples.append(example)

    print("The number of examples: ", count)

    random.shuffle(all_examples)
    train_examples = all_examples

    # tokenization
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    train_ids, train_length = convert_examples(train_examples, tokenizer)

    print("train max length: {:4}, {:4}, {:4}".format(*train_length))

    with open("./data/data.json", "w") as fw:
        json.dump({"train_ids": train_ids}, fw)


def train(bert_config, run_config, data_file):
    train_file = os.path.join(FLAGS.data_dir, data_file)
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
    # tf.logging.info("  Num dev examples = %d", len(dev_ids))
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
                                    output_dir=FLAGS.output_dir)]
                    )


def predict(bert_config, run_config, test_file):
    # processing test dataset
    all_examples = []
    with open(test_file, encoding="utf-8") as fr:
        for line in fr:
            data_line = json.loads(line)
            A = data_line["A"].split('\n')[2]
            B = data_line["B"].split('\n')[2]
            C = data_line["C"].split('\n')[2]

            example = Example(A=A, B=B, C=C)

            all_examples.append(example)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    test_ids, test_length = convert_examples(all_examples, tokenizer)

    test_features = convert_ids_to_features(token_ids=test_ids, tokenizer=tokenizer,
                                            max_seq_length=FLAGS.max_seq_length, is_training=True)

    # building estimator
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=None,
        num_warmup_steps=None,
        margin=FLAGS.margin)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        # train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.eval_batch_size)

    test_input_fn = input_fn_builder(input_features=test_features, seq_length=FLAGS.max_seq_length,
                                     is_training=False, drop_remainder=False)
    predictions = estimator.predict(test_input_fn, yield_single_examples=True)

    output_path = "../output/output.txt"
    if os.path.exists("./output") is False:
        os.mkdir("./output")
    ouf = open(output_path, "w", encoding="utf-8")
    for item in tqdm(predictions):
        pos_logit = item["pos_logit"]
        neg_logit = item["neg_logit"]
        if pos_logit > neg_logit:
            print("B", file=ouf)
        else:
            print("C", file=ouf)
    ouf.close()


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

    if FLAGS.do_pro:
        process()

    if FLAGS.do_train:
        train(bert_config, run_config, FLAGS.data_file)

    if FLAGS.do_eval:
        predict(bert_config, run_config, FLAGS.test_file)


if __name__ == "__main__":
    main()
