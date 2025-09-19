import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
 
block_size = 8
batch_size = 32
max_iters = 3000
eval_iters = 200
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#==============
torch.manual_seed(1337)

# Getting the dataset
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
response = requests.get(url)

# Save the content to a file
with open('input.txt', 'w') as file:
    file.write(response.text)

# print('The length of the dataset is', len(response.text))
# print('Text:', response.text[:1000])

# Getting the unique characters
chars = sorted(list(set(response.text)))
vocab_size = len(chars)
print(''.join(chars))
print('Vocab size:', vocab_size)

# Creating the mapping characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[j] for j in l])

# print(encode('hii there'))
# print(decode(encode('hii there')))


# Converting the dataset into a tensor
data = torch.tensor(encode(response.text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# Splitting the dataset
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print('Train data length:', len(train_data))
print('Val data length:', len(val_data))


def get_batch(split):
    # Generate a batch of data for training or validation
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t + 1]
#         target = yb[b, t]
#         print(f'When input is: {context.tolist()} the target is: {target}')


xb, yb = get_batch('train')
# print("Inputs:")
# print(xb.shape)
# print(xb)
# print('Targets:')
# print(yb.shape)
# print(yb)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets = None):
        # batch , time and channel
        # idx = batch of input tokens
        # targets = future token which we r trying to preict
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            # reshaping bcoz of pytorch it expects in other format 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss  = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # Generate new tokens
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Ensure idx is on the same device as the model
            idx = idx.to(self.token_embedding_table.weight.device)
            # get the predictions
            logits, loss = self(idx)
            # focusing only on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # applying the softmax to get the probabilites
            probs = F.softmax(logits, dim=-1)  #(B, C)
            # sample from the distributions
            # multinomail picks sample from the prob distribution
            idx_next = torch.multinomial(probs, num_samples=1)  #(B, 1)
            # appending the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        
        return idx

# Using the model
model = BigramLanguageModel(vocab_size)
m = model.to(device)


# Generating text
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val_loss{losses['val']:.4f}")
        
        # data 
        xb, yb = get_batch('train')
        
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

print(loss.item())

# Generating from model 
context = torch.zeros((1, 1), dtype=torch.long, device = device)
print((decode(m.generate(context, max_new_tokens = 600)[0].tolist())))