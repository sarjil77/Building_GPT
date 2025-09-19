"""
Configuration file for GPT training
"""

class GPTConfig:
    # Model hyperparameters
    batch_size = 64          # how many independent sequences will we process in parallel?
    block_size = 256         # what is the maximum context length for predictions?
    n_embd = 384            # embedding dimension
    n_head = 6              # number of attention heads
    n_layer = 6             # number of transformer blocks
    dropout = 0.2           # dropout probability
    
    # Training hyperparameters
    max_iters = 5000        # maximum training iterations
    eval_interval = 500     # how often to evaluate
    eval_iters = 200        # how many iterations to average for evaluation
    learning_rate = 3e-4    # learning rate for AdamW optimizer
    
    # Hardware
    device = 'cuda:2'       # device to run on ('cuda' for GPU, 'cpu' for CPU)
    
    # Data
    data_path = 'input.txt' # path to training data
    train_split = 0.9       # fraction of data to use for training
    
    # Generation
    max_new_tokens = 500    # maximum number of tokens to generate
    
    # Model saving/loading
    save_checkpoint = True
    checkpoint_path = 'checkpoints/'
    save_interval = 1000    # save checkpoint every N iterations
    
    # Logging
    log_interval = 100      # how often to log training progress
    log_file = 'training.log'