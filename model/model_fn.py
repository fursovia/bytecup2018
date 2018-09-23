import tensorflow as tf


def encoding_layer(rnn_inputs, params):

    stacked_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(params['rnn_size']), params['keep_prob']) for _ in range(params['num_layers'])]
    )

    outputs, state = tf.nn.dynamic_rnn(cell=stacked_cells,
                                       inputs=rnn_inputs,
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state,
                         dec_cell,
                         dec_embed_input,
                         target_sequence_length,
                         max_summary_length,
                         output_layer,
                         params):

    dec_cell = tf.contrib.rnn.DropoutWrapper(cell=dec_cell,
                                             output_keep_prob=params['keep_prob'])

    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                               sequence_length=target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                              helper=helper,
                                              initial_state=encoder_state,
                                              output_layer=output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer_infer(encoder_state,
                         dec_cell,
                         dec_embeddings,
                         max_summary_length,
                         output_layer,
                         params):

    dec_cell = tf.contrib.rnn.DropoutWrapper(cell=dec_cell,
                                             output_keep_prob=params['keep_prob'])

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,
                                                      start_tokens=tf.fill([params['batch_size']], params['go_id']),
                                                      end_token=params['end_id'])

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                              helper=helper,
                                              initial_state=encoder_state,
                                              output_layer=output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer(dec_input,
                   encoder_state,
                   target_sequence_length,
                   max_target_sequence_length,
                   embeddings,
                   params,
                   is_training):

    cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.LSTMCell(params['rnn_size']) for _ in range(params['num_layers'])]
    )

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(params['vocab_size'])
        train_output = decoding_layer_train(encoder_state,
                                            cells,
                                            dec_input,
                                            target_sequence_length,
                                            max_target_sequence_length,
                                            output_layer,
                                            params)

    if not is_training:
        with tf.variable_scope("decode", reuse=True):
            infer_output = decoding_layer_infer(encoder_state,
                                                cells,
                                                embeddings,
                                                max_target_sequence_length,
                                                output_layer,
                                                params)

        return train_output, infer_output
    else:
        return train_output, None


def seq2seq_model(input_data,
                  target_data,
                  target_sequence_length,
                  max_target_sentence_length,
                  embeddings,
                  params,
                  is_training):

    enc_outputs, enc_states = encoding_layer(input_data, params)

    train_output, infer_output = decoding_layer(target_data,
                                                enc_states,
                                                target_sequence_length,
                                                max_target_sentence_length,
                                                embeddings,
                                                params,
                                                is_training)

    return train_output, infer_output


def build_model(is_training, sentences, labels, params):

    if not is_training:
        max_summary_length = 15
    else:
        max_summary_length = tf.shape(labels)[1]

    # TODO: должен быть тензор из длин последовательностей
    target_sequence_length = tf.fill([params['batch_size']], max_summary_length)

    embeddings = tf.get_variable("embeddings",
                                 shape=[params['vocab_size'], params['embedding_size']],
                                 dtype=tf.float32)

    source_input = tf.nn.embedding_lookup(embeddings, sentences)
    if not is_training:
        target_input = source_input
    else:
        target_input = tf.nn.embedding_lookup(embeddings, labels)

    train_logits, inference_logits = seq2seq_model(source_input,
                                                   target_input,
                                                   target_sequence_length,
                                                   max_summary_length,
                                                   embeddings,
                                                   params,
                                                   is_training)

    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    return training_logits, inference_logits


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model'):
        training_logits, inference_logits = build_model(is_training, features, labels, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'preds': inference_logits}

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # TODO: переделать маску
    mask = tf.sequence_mask(tf.fill([params['batch_size']], tf.shape(features)[1]), tf.shape(features)[1], dtype=tf.float32, name='mask')
    # mask = tf.ones_like(labels, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(training_logits, labels, mask)

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
