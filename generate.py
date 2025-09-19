"""
Inference script for generating text with trained GPT model
"""

import torch
import argparse
from train_gpt import GPTLanguageModel, load_checkpoint, load_data
from config import GPTConfig

def generate_text(model, encode, decode, device, prompt="", max_tokens=500, temperature=1.0, top_k=None):
    """Generate text from the model"""
    model.eval()
    
    # Encode prompt if provided
    if prompt:
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
        generated_text = decode(generated[0].tolist())
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate text with GPT model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pth', help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=500, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter')
    parser.add_argument('--output', type=str, default=None, help='Output file to save generated text')
    
    args = parser.parse_args()
    
    # Load configuration and data
    config = GPTConfig()
    device = config.device if torch.cuda.is_available() and 'cuda' in config.device else 'cpu'
    
    # Load data for vocab info
    train_data, val_data, vocab_size, encode, decode = load_data(config.data_path)
    
    # Create and load model
    model = GPTLanguageModel(vocab_size, config)
    model = model.to(device)
    
    # Load checkpoint
    try:
        iter_num, loss = load_checkpoint(args.checkpoint, model)
        print(f"Loaded model from iteration {iter_num} with loss {loss:.4f}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Generate text
    print("Generating text...")
    generated_text = generate_text(
        model, encode, decode, device, 
        args.prompt, args.max_tokens, args.temperature, args.top_k
    )
    
    # Output
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(generated_text)
        print(f"Saved generated text to: {args.output}")

if __name__ == "__main__":
    main()