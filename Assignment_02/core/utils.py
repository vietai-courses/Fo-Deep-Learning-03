import numpy as np
import torch

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_case_rnn_step_forward(rnn_step_forward):
    N, D, H = 3, 10, 4

    x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
    prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
    Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
    Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
    b = np.linspace(-0.2, 0.4, num=H)

    next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
    expected_next_h = np.asarray([
    [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
    [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],
    [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])
    return expected_next_h, next_h

def test_case_rnn_step_backward(rnn_step_forward, rnn_step_backward, eval_numerical_gradient_array):
    np.random.seed(231)
    N, D, H = 4, 5, 6
    x = np.random.randn(N, D)
    h = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    out, cache = rnn_step_forward(x, h, Wx, Wh, b)

    dnext_h = np.random.randn(*out.shape)

    fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
    fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
    dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
    db_num = eval_numerical_gradient_array(fb, b, dnext_h)
    groud_truth = (dx_num, dprev_h_num, dWx_num, dWh_num, db_num)

    dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
    predict = (dx, dprev_h, dWx, dWh, db)
    return groud_truth, predict

def test_case_rnn_forward(rnn_forward):
    N, T, D, H = 2, 3, 4, 5

    x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)
    h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
    Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
    b = np.linspace(-0.7, 0.1, num=H)

    h, _ = rnn_forward(x, h0, Wx, Wh, b)
    expected_h = np.asarray([
    [
        [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
        [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
        [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
    ],
    [
        [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
        [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
        [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])
    return expected_h, h

def test_case_rnn_backward(rnn_forward, rnn_backward, eval_numerical_gradient_array):
    np.random.seed(231)
    N, D, T, H = 2, 3, 10, 5

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(H)

    out, cache = rnn_forward(x, h0, Wx, Wh, b)

    dout = np.random.randn(*out.shape)

    dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
    predict = (dx, dh0, dWx, dWh, db)

    fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
    fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
    fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
    fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
    fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)
    groud_truth = (dx_num, dh0_num, dWx_num, dWh_num, db_num)
    return groud_truth, predict

def test_case_word_embedding_forward(word_embedding_forward):
    N, T, V, D = 2, 4, 5, 3

    x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
    W = np.linspace(0, 1, num=V*D).reshape(V, D)

    out, _ = word_embedding_forward(x, W)
    expected_out = np.asarray([
    [[ 0.,          0.07142857,  0.14285714],
    [ 0.64285714,  0.71428571,  0.78571429],
    [ 0.21428571,  0.28571429,  0.35714286],
    [ 0.42857143,  0.5,         0.57142857]],
    [[ 0.42857143,  0.5,         0.57142857],
    [ 0.21428571,  0.28571429,  0.35714286],
    [ 0.,          0.07142857,  0.14285714],
    [ 0.64285714,  0.71428571,  0.78571429]]])
    return expected_out, out

def test_case_word_embedding_backward(word_embedding_forward, word_embedding_backward, eval_numerical_gradient_array):
    np.random.seed(231)

    N, T, V, D = 50, 3, 5, 6
    x = np.random.randint(V, size=(N, T))
    W = np.random.randn(V, D)

    out, cache = word_embedding_forward(x, W)
    dout = np.random.randn(*out.shape)
    dW = word_embedding_backward(dout, cache)

    f = lambda W: word_embedding_forward(x, W)[0]
    dW_num = eval_numerical_gradient_array(f, W, dout)
    return dW, dW_num

def test_case_lstm_step_forward(lstm_step_forward):
    N, D, H = 3, 4, 5
    x = np.linspace(-0.4, 1.2, num=N*D).reshape(N, D)
    prev_h = np.linspace(-0.3, 0.7, num=N*H).reshape(N, H)
    prev_c = np.linspace(-0.4, 0.9, num=N*H).reshape(N, H)
    Wx = np.linspace(-2.1, 1.3, num=4*D*H).reshape(D, 4 * H)
    Wh = np.linspace(-0.7, 2.2, num=4*H*H).reshape(H, 4 * H)
    b = np.linspace(0.3, 0.7, num=4*H)

    next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)

    expected_next_h = np.asarray([
        [ 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904],
        [ 0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ],
        [ 0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299  ]])
    expected_next_c = np.asarray([
        [ 0.32986176,  0.39145139,  0.451556,    0.51014116,  0.56717407],
        [ 0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345],
        [ 0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676]])

    return expected_next_h, next_h, expected_next_c, next_c

def test_case_lstm_forward(lstm_forward):
    N, D, H, T = 2, 5, 4, 3
    x = np.linspace(-0.4, 0.6, num=N*T*D).reshape(N, T, D)
    h0 = np.linspace(-0.4, 0.8, num=N*H).reshape(N, H)
    Wx = np.linspace(-0.2, 0.9, num=4*D*H).reshape(D, 4 * H)
    Wh = np.linspace(-0.3, 0.6, num=4*H*H).reshape(H, 4 * H)
    b = np.linspace(0.2, 0.7, num=4*H)

    h, cache = lstm_forward(x, h0, Wx, Wh, b)

    expected_h = np.asarray([
    [[ 0.01764008,  0.01823233,  0.01882671,  0.0194232 ],
      [ 0.11287491,  0.12146228,  0.13018446,  0.13902939],
      [ 0.31358768,  0.33338627,  0.35304453,  0.37250975]],
    [[ 0.45767879,  0.4761092,   0.4936887,   0.51041945],
      [ 0.6704845,   0.69350089,  0.71486014,  0.7346449 ],
      [ 0.81733511,  0.83677871,  0.85403753,  0.86935314]]])

    return expected_h, h

def test_case_lstm_backward(lstm_forward, lstm_backward, eval_numerical_gradient_array):
    np.random.seed(231)

    N, D, T, H = 2, 3, 10, 6

    x = np.random.randn(N, T, D)
    h0 = np.random.randn(N, H)
    Wx = np.random.randn(D, 4 * H)
    Wh = np.random.randn(H, 4 * H)
    b = np.random.randn(4 * H)

    out, cache = lstm_forward(x, h0, Wx, Wh, b)

    dout = np.random.randn(*out.shape)

    dx, dh0, dWx, dWh, db = lstm_backward(dout, cache)

    fx = lambda x: lstm_forward(x, h0, Wx, Wh, b)[0]
    fh0 = lambda h0: lstm_forward(x, h0, Wx, Wh, b)[0]
    fWx = lambda Wx: lstm_forward(x, h0, Wx, Wh, b)[0]
    fWh = lambda Wh: lstm_forward(x, h0, Wx, Wh, b)[0]
    fb = lambda b: lstm_forward(x, h0, Wx, Wh, b)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
    dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
    dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
    db_num = eval_numerical_gradient_array(fb, b, dout)

    return dx_num, dx, dh0_num, dh0, dWx_num, dWx, dWh_num, dWh, db_num, db

def test_case_loss_RNN_captioning(CaptioningRNN):
    N, D, W, H = 10, 20, 30, 40
    word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
    V = len(word_to_idx)
    T = 13

    model = CaptioningRNN(
        word_to_idx,
        input_dim=D,
        wordvec_dim=W,
        hidden_dim=H,
        cell_type='rnn',
        dtype=np.float64
    )

    # Set all model parameters to fixed values
    for k, v in model.params.items():
        model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

    features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
    captions = (np.arange(N * T) % V).reshape(N, T)

    loss, grads = model.loss(features, captions)
    
    expected_loss = 9.83235591003
    return loss, expected_loss

def test_case_loss_LSTM_captioning(CaptioningRNN):
  N, D, W, H = 10, 20, 30, 40
  word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
  V = len(word_to_idx)
  T = 13

  model = CaptioningRNN(
      word_to_idx,
      input_dim=D,
      wordvec_dim=W,
      hidden_dim=H,
      cell_type='lstm',
      dtype=np.float64
  )

  # Set all model parameters to fixed values
  for k, v in model.params.items():
    model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

  features = np.linspace(-0.5, 1.7, num=N*D).reshape(N, D)
  captions = (np.arange(N * T) % V).reshape(N, T)

  loss, grads = model.loss(features, captions)
  expected_loss = 9.82445935443
  return loss, expected_loss

def test_case_gradient_CaptioningRNN(CaptioningRNN):
    np.random.seed(231)

    batch_size = 2
    timesteps = 3
    input_dim = 4
    wordvec_dim = 5
    hidden_dim = 6
    word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
    vocab_size = len(word_to_idx)

    captions = np.random.randint(vocab_size, size=(batch_size, timesteps))
    features = np.random.randn(batch_size, input_dim)

    model = CaptioningRNN(
        word_to_idx,
        input_dim=input_dim,
        wordvec_dim=wordvec_dim,
        hidden_dim=hidden_dim,
        cell_type='rnn',
        dtype=np.float64,
    )

    loss, grads = model.loss(features, captions)
    return model, loss, grads

def test_case_multihead_attention(MultiHeadAttention):
  torch.manual_seed(231)

  # Choose dimensions such that they are all unique for easier debugging:
  # Specifically, the following values correspond to N=1, H=2, T=3, E//H=4, and E=8.
  batch_size = 1
  sequence_length = 3
  embed_dim = 8
  attn = MultiHeadAttention(embed_dim, num_heads=2)

  # Self-attention.
  data = torch.randn(batch_size, sequence_length, embed_dim)
  self_attn_output = attn(query=data, key=data, value=data)

  # Masked self-attention.
  mask = torch.randn(sequence_length, sequence_length) < 0.5
  masked_self_attn_output = attn(query=data, key=data, value=data, attn_mask=mask)

  # Attention using two inputs.
  other_data = torch.randn(batch_size, sequence_length, embed_dim)
  attn_output = attn(query=data, key=other_data, value=other_data)

  expected_self_attn_output = np.asarray([[
  [-0.2494,  0.1396,  0.4323, -0.2411, -0.1547,  0.2329, -0.1936,
            -0.1444],
          [-0.1997,  0.1746,  0.7377, -0.3549, -0.2657,  0.2693, -0.2541,
            -0.2476],
          [-0.0625,  0.1503,  0.7572, -0.3974, -0.1681,  0.2168, -0.2478,
            -0.3038]]])

  expected_masked_self_attn_output = np.asarray([[
  [-0.1347,  0.1934,  0.8628, -0.4903, -0.2614,  0.2798, -0.2586,
            -0.3019],
          [-0.1013,  0.3111,  0.5783, -0.3248, -0.3842,  0.1482, -0.3628,
            -0.1496],
          [-0.2071,  0.1669,  0.7097, -0.3152, -0.3136,  0.2520, -0.2774,
            -0.2208]]])

  expected_attn_output = np.asarray([[
  [-0.1980,  0.4083,  0.1968, -0.3477,  0.0321,  0.4258, -0.8972,
            -0.2744],
          [-0.1603,  0.4155,  0.2295, -0.3485, -0.0341,  0.3929, -0.8248,
            -0.2767],
          [-0.0908,  0.4113,  0.3017, -0.3539, -0.1020,  0.3784, -0.7189,
            -0.2912]]])

  return expected_self_attn_output, self_attn_output, expected_masked_self_attn_output, masked_self_attn_output, expected_attn_output, attn_output

def test_case_positional_encoding(PositionalEncoding):
  torch.manual_seed(231)

  batch_size = 1
  sequence_length = 2
  embed_dim = 6
  data = torch.randn(batch_size, sequence_length, embed_dim)

  pos_encoder = PositionalEncoding(embed_dim)
  output = pos_encoder(data)

  expected_pe_output = np.asarray([[[-1.2340,  1.1127,  1.6978, -0.0865, -0.0000,  1.2728],
                                    [ 0.9028, -0.4781,  0.5535,  0.8133,  1.2644,  1.7034]]])
  return expected_pe_output, output

def test_case_transformer_captioning(CaptioningTransformer):
  torch.manual_seed(231)
  np.random.seed(231)

  N, D, W = 4, 20, 30
  word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
  V = len(word_to_idx)
  T = 3

  transformer = CaptioningTransformer(
      word_to_idx,
      input_dim=D,
      wordvec_dim=W,
      num_heads=2,
      num_layers=2,
      max_length=30
  )

  # Set all model parameters to fixed values
  for p in transformer.parameters():
      p.data = torch.tensor(np.linspace(-1.4, 1.3, num=p.numel()).reshape(*p.shape))

  features = torch.tensor(np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D))
  captions = torch.tensor((np.arange(N * T) % V).reshape(N, T))

  scores = transformer(features, captions)
  expected_scores = np.asarray([[[-16.9532,   4.8261,  26.6054],
          [-17.1033,   4.6906,  26.4844],
          [-15.0708,   4.1108,  23.2924]],
          [[-17.1767,   4.5897,  26.3562],
          [-15.6017,   4.8693,  25.3403],
          [-15.1028,   4.6905,  24.4839]],
          [[-17.2172,   4.7701,  26.7574],
          [-16.6755,   4.8500,  26.3754],
          [-17.2172,   4.7701,  26.7574]],
          [[-16.3669,   4.1602,  24.6872],
          [-16.7897,   4.3467,  25.4831],
          [-17.0103,   4.7775,  26.5652]]])

  return expected_scores, scores