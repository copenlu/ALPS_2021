ALL_ARGUMENTS = {
    # Common arguments
    "gpu": True,
    "init_only": False,  # Whether to train the model
    "seed": 73,  # Random seed
    "labels": 3,  # Number of labels
    "model_path": 'tweet_model',  # Path where the model will be serialized
    "batch_size": 7,  # Batch size
    "mode": 'train',  # Mode for the script = train/test
    "model": 'cnn',  # Model for training lstm/cnn/transformer
}

CNN_ARGS = {
    "lr": 0.001,
    "batch_size": 64,
    "dropout": 0.05,
    "kernel_heights": [2, 3, 4],
    "out_channels": 50,
    "embedding_dim": 300,
    "in_channels": 1,
    "stride": 1,
    "padding": 0,
    "epochs": 3,
    "activation": 'relu',  # tanh, sigmoid
    "pooling": 'max'  # max, average
}

LSTM_ARGS = {
    "batch_size": 16,
    "lr": 0.001,
    "dropout": 0.05,
    "hidden_lstm": 100,
    "num_layers": 1,
    "hidden_sizes": [200, 100],
    "epochs": 3,
    "bidirectional": True,
    "embedding_dim": 200,
}

TRANSFORMER_ARGS = {
    "lr": 3e-5,
    "batch_size": 8,
    "epochs": 5
}


def get_model_args(model_type):
    if model_type == 'cnn':
        return CNN_ARGS
    elif model_type == 'lstm':
        return LSTM_ARGS
    elif model_type == 'transformer':
        return TRANSFORMER_ARGS
