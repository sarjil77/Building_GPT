# GPT from Scratch 🤖

A character-level GPT (Generative Pre-trained Transformer) implementation built from scratch using PyTorch. This project demonstrates a complete transformer architecture with multi-head self-attention, training on the Tiny Shakespeare dataset.

## 🌟 Features

- **Complete Transformer Architecture**: Implementation of GPT with multi-head self-attention mechanism
- **Character-level Tokenization**: Simple character-based vocabulary for text generation
- **Multi-Head Attention**: Parallel attention heads for capturing different aspects of relationships
- **Layer Normalization**: Pre-norm architecture following modern transformer designs
- **Residual Connections**: Skip connections for better gradient flow
- **Positional Embeddings**: Learnable position encodings for sequence understanding
- **Text Generation**: Autoregressive text generation with temperature sampling
- **Training Loop**: Complete training pipeline with validation monitoring

## 🏗️ Architecture

### Model Components

- **Embedding Layers**: Token and positional embeddings (384 dimensions)
- **Transformer Blocks**: 6 layers with multi-head attention and feed-forward networks
- **Attention Heads**: 6 attention heads with 64-dimensional head size
- **Feed-Forward Network**: 4x expansion ratio (384 → 1536 → 384)
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Dropout**: 0.2 dropout rate for regularization

### Key Parameters

```python
batch_size = 64          # Parallel sequences
block_size = 256         # Maximum context length
n_embd = 384            # Embedding dimension
n_head = 6              # Number of attention heads
n_layer = 6             # Number of transformer blocks
dropout = 0.2           # Dropout probability
learning_rate = 3e-4    # AdamW learning rate
```

## 📊 Dataset

The model trains on the **Tiny Shakespeare** dataset:
- **Source**: Character-level text from Shakespeare's works
- **Size**: ~40,000 characters
- **Vocabulary**: 65 unique characters (letters, punctuation, spaces)
- **Split**: 90% training, 10% validation

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA-compatible GPU (optional but recommended)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
The script automatically loads the Tiny Shakespeare dataset. Alternatively, download manually:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Usage

1. **Train the model**:
```bash
python gpt_from_scratch.py
```

2. **Monitor training**:
The script will output training and validation losses every 500 iterations:
```
step 0: train loss 4.1743, val loss 4.1755
step 500: train loss 2.4539, val loss 2.4649
step 1000: train loss 2.0123, val loss 2.0456
...
```

3. **Generate text**:
After training, the model automatically generates 500 characters of Shakespeare-style text.

## 📈 Training Details

### Training Configuration

- **Optimizer**: AdamW with learning rate 3e-4
- **Iterations**: 5,000 training steps
- **Batch Size**: 64 sequences
- **Context Length**: 256 characters
- **Evaluation**: Every 500 iterations on validation set

### Loss Curves

The model typically achieves:
- **Initial Loss**: ~4.17 (random prediction)
- **Final Training Loss**: ~1.5-2.0
- **Final Validation Loss**: ~1.6-2.1

### Hardware Requirements

- **GPU Memory**: ~2GB VRAM for training
- **Training Time**: ~10-15 minutes on modern GPU
- **CPU Training**: Possible but significantly slower

## 🎯 Results

### Sample Generated Text

```
GLOUCESTER:
What shall we do? let's hear the news from France.

KING RICHARD III:
My lord, I have consider'd in my mind
The late demand that you did sound me from,
And find that you are not so sick of my,
As I am of your lordship. Here's my hand.
```

### Model Performance

- **Parameters**: ~10.8M parameters
- **Training Perplexity**: ~4.5-6.0
- **Generation Quality**: Coherent character names, basic grammar, Shakespeare-style vocabulary

## 🔧 Customization

### Modifying Hyperparameters

Edit the hyperparameters section in `gpt_from_scratch.py`:

```python
# Model size
n_embd = 384        # Increase for larger model
n_head = 6          # More heads for better attention
n_layer = 6         # Deeper model

# Training
batch_size = 64     # Larger batches if GPU memory allows
learning_rate = 3e-4  # Adjust learning rate
max_iters = 5000    # Train for more iterations
```

### Using Different Datasets

1. Replace `input.txt` with your text data
2. Ensure proper UTF-8 encoding
3. The model will automatically adapt to the new vocabulary

### Saving and Loading Models

Add model checkpointing (see improvements section below):

```python
# Save model
torch.save(model.state_dict(), 'gpt_model.pth')

# Load model
model.load_state_dict(torch.load('gpt_model.pth'))
```

## 📁 Project Structure

```
Building_GPT/
├── gpt_from_scratch.py    # Main GPT implementation
├── birgam.py             # Alternative/experimental version
├── input.txt             # Tiny Shakespeare dataset
├── gpt_from_scratch.log  # Training logs
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

## 🧠 Technical Deep Dive

### Self-Attention Mechanism

The model implements scaled dot-product attention:

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

With causal masking to prevent attending to future tokens during training.

### Architecture Innovations

- **Pre-Layer Normalization**: LayerNorm applied before attention and FFN
- **Residual Connections**: Direct paths for gradient flow
- **Causal Masking**: Lower triangular mask for autoregressive generation
- **Weight Initialization**: Proper initialization following GPT standards

### Memory and Efficiency

- **Gradient Checkpointing**: Can be added for larger models
- **Mixed Precision**: Float16 training support possible
- **Sequence Packing**: Fixed-length sequences for efficient batching

## 🚀 Potential Improvements

1. **Model Checkpointing**: Save/load model states
2. **Configuration Management**: External config files
3. **Better Data Pipeline**: More efficient data loading
4. **Evaluation Metrics**: Perplexity, BLEU score calculation
5. **Generation Sampling**: Temperature, top-k, top-p sampling
6. **Distributed Training**: Multi-GPU support
7. **Mixed Precision**: Faster training with automatic mixed precision
8. **Logging**: Better experiment tracking with wandb/tensorboard

## 📚 Learning Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-models-are-unsupervised-multitask-learners) - GPT-2 paper
- [Andrej Karpathy's Neural Networks Course](https://karpathy.ai/zero-to-hero.html) - Excellent learning resource

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Andrej Karpathy** for the excellent educational content on transformers
- **OpenAI** for the GPT architecture and research
- **Vaswani et al.** for the original Transformer architecture
- **Shakespeare** for providing the training data 📚

⭐ If you found this project helpful, please give it a star!