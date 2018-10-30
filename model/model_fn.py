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
        self.beam_width = 3
        self.max_summary_length = 15
        self.rnn_cell = tf.nn.rnn_cell.LSTMCell
        self.embed_matrix = tf.get_variable("embedding_matrix",
                                            shape=[self.vocab_size, self.embedding_size],
                                            dtype=tf.float32)

        with tf.variable_scope('decoder/projection'):
            self.output_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

    def embed_inputs(self, inputs):
        return tf.nn.embedding_lookup(self.embed_matrix, inputs)

    def seq2seq_model(self,
                      input_data,
                      target_data,
                      is_training):

        enc_outputs, enc_state = self.encoding_layer(input_data)

        output = self.decoding_layer(target_data,
                                     enc_state,
                                     enc_outputs,
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
        return outputs, state[0]

    def decoding_layer(self,
                       dec_input,
                       encoder_state,
                       enc_outputs,
                       is_training):

        cells = self.rnn_cell(self.rnn_size)
        # cells = tf.nn.rnn_cell.MultiRNNCell([dec_cell for _ in range(self.num_layers)])

        with tf.name_scope('decoder'), tf.variable_scope('decoder') as decoder_scope:
            if is_training:
                output = self.decoding_layer_train(encoder_state,
                                                   enc_outputs,
                                                   cells,
                                                   dec_input,
                                                   decoder_scope)
            else:
                # return encoder_state, enc_outputs, cells, decoder_scope
                output = self.decoding_layer_infer(encoder_state,
                                                   enc_outputs,
                                                   cells,
                                                   decoder_scope)

        return output

    def decoding_layer_train(self,
                             encoder_state,
                             enc_outputs,
                             dec_cell,
                             dec_embed_input,
                             scope):

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.rnn_size * 2,
                                                                   enc_outputs,
                                                                   memory_sequence_length=self.target_sequence_length,
                                                                   normalize=True)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                           attention_mechanism,
                                                           attention_layer_size=self.rnn_size * 2)

        initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)

        helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=self.target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                  helper=helper,
                                                  initial_state=initial_state,
                                                  output_layer=self.output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                          scope=scope)
        return outputs

    def decoding_layer_infer(self,
                             encoder_state,
                             enc_outputs,
                             dec_cell,
                             scope):

        tiled_encoder_output = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=self.beam_width)
        tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_width)
        tiled_seq_len = tf.contrib.seq2seq.tile_batch(tf.fill([self.batch_size], self.max_summary_length),
                                                      multiplier=self.beam_width)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.rnn_size * 2,
                                                                   tiled_encoder_output,
                                                                   memory_sequence_length=tiled_seq_len,
                                                                   normalize=True)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                           attention_mechanism,
                                                           attention_layer_size=self.rnn_size * 2)

        initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
        initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=self.embed_matrix,
            start_tokens=tf.fill([self.batch_size], self.go_id),
            end_token=self.end_id,
            initial_state=initial_state,
            beam_width=self.beam_width,
            output_layer=self.output_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                          # impute_finished=True,
                                                          maximum_iterations=self.max_summary_length,
                                                          scope=scope)

        return outputs

    def build_model(self, is_training, features, target):

        if is_training:
            self.max_summary_length = tf.shape(target)[1]

        self.target_sequence_length = tf.fill([self.batch_size], self.max_summary_length)

        source_input = self.embed_inputs(features['source'])
        if is_training:
            target_input = self.embed_inputs(target)
        else:
            target_input = None

        logits = self.seq2seq_model(source_input,
                                    target_input,
                                    is_training)

        if is_training:
            logits = tf.identity(logits.rnn_output, name='logits')
        else:
            logits = tf.identity(logits.predicted_ids[:, :, 1], name='logits')
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
