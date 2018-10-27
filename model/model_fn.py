import tensorflow as tf


class Model:
    def __init__(self, params):
        self.rnn_size = params['rnn_size']
        self.keep_prob = params['keep_prob']
        self.num_layers = params['num_layers']
        self.batch_size = params['batch_size']
        self.go_id = params['go_id']
        self.end_id = params['end_id']
        self.vocab_size = params['vocab_size']
        self.embedding_size = params['embedding_size']
        self.max_summary_length = 15

        self.rnn_cell = tf.nn.rnn_cell.LSTMCell
        self.embed_matrix = tf.get_variable("embedding_matrix",
                                            shape=[self.vocab_size, self.embedding_size],
                                            dtype=tf.float32)

    def embed_inputs(self, inputs):
        return tf.nn.embedding_lookup(self.embed_matrix, inputs)

    def seq2seq_model(self,
                      input_data,
                      target_data,
                      target_sequence_length,
                      max_target_sentence_length,
                      is_training):

        enc_outputs, enc_states = self.encoding_layer(input_data)

        output = self.decoding_layer(target_data,
                                     enc_states,
                                     target_sequence_length,
                                     max_target_sentence_length,
                                     is_training)

        return output

    def encoding_layer(self, inputs):
        # TODO: variational dropout
        enc_cell = self.rnn_cell(self.rnn_size)

        stacked_cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(enc_cell, self.keep_prob) for _ in range(self.num_layers)])

        outputs, state = tf.nn.dynamic_rnn(cell=stacked_cells,
                                           inputs=inputs,
                                           dtype=tf.float32)
        return outputs, state

    def decoding_layer(self, dec_input,
                       encoder_state,
                       target_sequence_length,
                       max_target_sequence_length,
                       is_training):

        with tf.variable_scope('decoder/projection'):
            output_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

        dec_cell = self.rnn_cell(self.rnn_size)
        cells = tf.nn.rnn_cell.MultiRNNCell([dec_cell for _ in range(self.num_layers)])

        with tf.name_scope('decoder'), tf.variable_scope('decoder') as decoder_scope:
            if is_training:
                output = self.decoding_layer_train(encoder_state,
                                                   cells,
                                                   dec_input,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   output_layer,
                                                   decoder_scope)
            else:
                output = self.decoding_layer_infer(encoder_state,
                                                   cells,
                                                   max_target_sequence_length,
                                                   output_layer,
                                                   decoder_scope)

        return output

    def decoding_layer_train(self,
                             encoder_state,
                             dec_cell,
                             dec_embed_input,
                             target_sequence_length,
                             max_summary_length,
                             output_layer,
                             scope):

        dec_cell = tf.nn.rnn_cell.DropoutWrapper(cell=dec_cell, output_keep_prob=self.keep_prob)

        helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                  helper=helper,
                                                  initial_state=encoder_state,
                                                  output_layer=output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=max_summary_length,
                                                          scope=scope)
        return outputs

    def decoding_layer_infer(self,
                             encoder_state,
                             dec_cell,
                             max_summary_length,
                             output_layer,
                             scope):

        dec_cell = tf.contrib.rnn.DropoutWrapper(cell=dec_cell,
                                                 output_keep_prob=self.keep_prob)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embed_matrix,
                                                          start_tokens=tf.fill([self.batch_size], self.go_id),
                                                          end_token=self.end_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                  helper=helper,
                                                  initial_state=encoder_state,
                                                  output_layer=output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=max_summary_length,
                                                          scope=scope)
        return outputs

    def build_model(self, is_training, features, target):

        if is_training:
            max_summary_length = tf.shape(target)[1]
        else:
            max_summary_length = self.max_summary_length

        # TODO: должен быть тензор из длин последовательностей
        target_sequence_length = tf.fill([self.batch_size], max_summary_length)

        source_input = self.embed_inputs(features['source'])
        # костыль. estimator.predict меняет таргет на None
        if is_training:
            target_input = self.embed_inputs(target)
        else:
            target_input = source_input

        logits = self.seq2seq_model(source_input,
                                    target_input,
                                    target_sequence_length,
                                    max_summary_length,
                                    is_training)

        logits = tf.identity(logits.rnn_output, name='logits')
        return logits


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model'):
        model = Model(params=params)
        logits = model.build_model(is_training, features, labels)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'preds': logits}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    target_sequence_length = tf.squeeze(features['target_len'])
    mask = tf.sequence_mask(target_sequence_length, tf.reduce_max(target_sequence_length),
                            dtype=tf.float32,
                            name='mask')
    # tf.nn.sparse_softmax_cross_entropy_with_logits
    # loss = tf.contrib.seq2seq.sequence_loss(logits, labels, mask)

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_sum(crossent * mask) / tf.to_float(params['batch_size'])

    tf.summary.scalar('loss', loss)

    optimizer_fn = tf.train.AdamOptimizer(params['learning_rate'])

    global_step = tf.train.get_global_step()

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params['learning_rate'],
        optimizer=optimizer_fn,
        name='optimizer')

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
