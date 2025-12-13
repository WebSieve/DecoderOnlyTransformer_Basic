class Training_config:
    """
    Configurations for training
    """

    # Model parameters
    # vocab_size = 50000
    max_seq_len = 256
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    intermediate_dim = 2048
    dropout_rate = 0.1
    stride = 1

    # Training parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    warmup_steps = 1000

    # Data
    data_path = "data/train.txt"

    # Checkpointing
    checkpoint_dir = r"I ain't giving out my directory!!!"
    save_every = 50

    # Device
    device = "cpu"  # or you could use cuda.

    tokenizer_name = "gpt2"
