import os
from copy import copy

import tensorflow as tf
from tqdm import tqdm

import modeling
import optimization

if tf.__version__ >= "1.13.0":
    import tensorflow_estimator as tfes
else:
    tfes = tf

from tensorflow.contrib import tpu
from tensorflow.python.training.training_util import _get_or_create_global_step_read as get_global_step
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook


class Features:
    def __init__(self, pos_input_ids, pos_input_mask, pos_segment_ids,
                 neg_input_ids, neg_input_mask, neg_segment_ids):
        self.pos_input_ids = pos_input_ids
        self.pos_input_mask = pos_input_mask
        self.pos_segment_ids = pos_segment_ids
        self.neg_input_ids = neg_input_ids
        self.neg_input_mask = neg_input_mask
        self.neg_segment_ids = neg_segment_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def combine_and_padding(sent1_ids, sent2_ids, max_seq_length, delimiter):
    input_ids = []
    segment_ids = []

    input_ids.append(delimiter[0])
    segment_ids.append(0)

    input_ids += sent1_ids
    segment_ids += [0] * len(sent1_ids)
    input_ids.append(delimiter[1])
    segment_ids.append(0)

    input_ids += sent2_ids
    segment_ids += [1] * len(sent2_ids)
    input_ids.append(delimiter[1])
    segment_ids.append(1)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_ids_to_features(token_ids, tokenizer, max_seq_length, is_training):
    delimiter = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

    features = []
    for each_token_ids in token_ids:
        sent12_ids = each_token_ids[0]
        sent13_ids = copy(each_token_ids[0])
        sent2_ids = each_token_ids[1]
        sent3_ids = each_token_ids[2] if is_training else None

        _truncate_seq_pair(sent12_ids, sent2_ids, max_seq_length - 3)
        if is_training:
            _truncate_seq_pair(sent13_ids, sent3_ids, max_seq_length - 3)

        pos_features = combine_and_padding(sent12_ids, sent2_ids, max_seq_length, delimiter)
        neg_features = pos_features
        if is_training:
            neg_features = combine_and_padding(sent13_ids, sent3_ids, max_seq_length, delimiter)

        features.append(Features(pos_input_ids=pos_features[0],
                                 pos_input_mask=pos_features[1],
                                 pos_segment_ids=pos_features[2],
                                 neg_input_ids=neg_features[0],
                                 neg_input_mask=neg_features[1],
                                 neg_segment_ids=neg_features[2]))

    return features


def create_model_or_use_model(bert_config, is_training, input_ids, input_mask, segment_ids):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights_v2", [1, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        # output_bias = tf.get_variable(
        #     "output_bias_v2", [1], initializer=tf.zeros_initializer())

        with tf.variable_scope("logits"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            # logits = tf.nn.bias_add(logits, output_bias)
            # log_probs = tf.nn.sigmoid(logits)

        return logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, margin):

    def model_fn(features, labels, mode, params):

        pos_input_ids = features["pos_input_ids"]
        pos_input_mask = features["pos_input_mask"]
        pos_segment_ids = features["pos_segment_ids"]

        neg_input_ids = features["neg_input_ids"]
        neg_input_mask = features["neg_input_mask"]
        neg_segment_ids = features["neg_segment_ids"]

        is_training = (mode == tfes.estimator.ModeKeys.TRAIN)

        pos_logits = create_model_or_use_model(bert_config=bert_config, is_training=is_training,
                                               input_ids=pos_input_ids,
                                               input_mask=pos_input_mask,
                                               segment_ids=pos_segment_ids)
        neg_logits = create_model_or_use_model(bert_config=bert_config, is_training=is_training,
                                               input_ids=neg_input_ids,
                                               input_mask=neg_input_mask,
                                               segment_ids=neg_segment_ids)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name[6:] in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        if mode == tfes.estimator.ModeKeys.TRAIN:
            original_loss = tf.nn.relu(margin - pos_logits + neg_logits)
            total_loss = tf.reduce_mean(original_loss)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tfes.estimator.ModeKeys.PREDICT:
            predictions = {"pos_logit": pos_logits, "neg_logit": neg_logits}
            output_spec = tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % mode)

        return output_spec

    return model_fn


def input_fn_builder(input_features, seq_length, is_training, drop_remainder):
    all_pos_input_ids = []
    all_pos_input_mask = []
    all_pos_segment_ids = []
    all_neg_input_ids = []
    all_neg_input_mask = []
    all_neg_segment_ids = []

    for feature in input_features:
        all_pos_input_ids.append(feature.pos_input_ids)
        all_pos_input_mask.append(feature.pos_input_mask)
        all_pos_segment_ids.append(feature.pos_segment_ids)
        all_neg_input_ids.append(feature.neg_input_ids)
        all_neg_input_mask.append(feature.neg_input_mask)
        all_neg_segment_ids.append(feature.neg_segment_ids)

    # print("============================================================")
    # print("features processing")

    def input_fn(params):
        batch_size = params["batch_size"]
        num_examples = len(input_features)

        d = tf.data.Dataset.from_tensor_slices({
            "pos_input_ids": tf.constant(all_pos_input_ids,  shape=[num_examples, seq_length], dtype=tf.int32),
            "pos_input_mask": tf.constant(all_pos_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "pos_segment_ids": tf.constant(all_pos_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),

            "neg_input_ids": tf.constant(all_neg_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "neg_input_mask": tf.constant(all_neg_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "neg_segment_ids": tf.constant(all_neg_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


class EvalHook(SessionRunHook):
    def __init__(self, estimator,
                 eval_features, max_seq_length, eval_steps,
                 save_model_dir, th, output_dir):
        self.estimator = estimator
        self.eval_features = eval_features
        self.max_seq_length = max_seq_length
        self.eval_steps = eval_steps
        self.save_model_dir = save_model_dir
        self.th = th
        self.output_dir = output_dir

        if os.path.exists(self.save_model_dir) is False:
            os.mkdir(self.save_model_dir)
        self._timer = SecondOrStepTimer(every_steps=eval_steps)
        self._steps_per_run = 1
        self._global_step_tensor = None

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        # self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = get_global_step()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CheckpointSaverHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        # print(run_values.results)
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                metrics = self.evaluation(global_step)
                # print("================", MAP, MRR, self.th, type(MAP), type(MRR), type(self.th))
                if metrics["acc"] * 100 > self.th:
                    # print("================", MAP, MRR)
                    self._save(run_context.session, global_step, metrics)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            metrics = self.evaluation(last_step)
            if metrics["acc"] * 100 > self.th:
                self._save(session, last_step, metrics)

    def evaluation(self, global_step):
        dev_input_fn = input_fn_builder(input_features=self.eval_features, seq_length=self.max_seq_length,
                                        is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(dev_input_fn, yield_single_examples=True)

        results = []
        logits = []
        for item in tqdm(predictions):
            pos_logit = item["pos_logit"]
            neg_logit = item["neg_logit"]
            logits.append((pos_logit, neg_logit))
            results.append(1 if pos_logit > neg_logit else 0)

        acc = sum(results) / len(results)
        print(f"global_step: {global_step}, acc: {acc}")
        return {"acc": acc}

    def _save(self, session, step, metrics):
        save_file = os.path.join(self.save_model_dir, "step{}_acc{:5.4f}".format(step, metrics["acc"]))
        list_name = os.listdir(self.org_dir)
        for name in list_name:
            if "model.ckpt-{}".format(step-1) in name:
                org_name = os.path.join(self.output_dir, name)
                tag_name = save_file + "." + name.split(".")[-1]
                print("save {} to {}".format(org_name, tag_name))
                with open(org_name, "rb") as fr, open(tag_name, 'wb') as fw:
                    fw.write(fr.read())
