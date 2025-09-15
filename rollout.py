import math, torch, types, copy, re
import numpy as np
from torch.nn import functional as F

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096"
# args.n_layer = 12
# args.n_embd = 768
args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.4b-20250905-ctx4096"
args.n_layer = 24
args.n_embd = 1024
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-1.5b-20250429-ctx4096"
# args.n_layer = 24
# args.n_embd = 2048
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-2.9b-20250519-ctx4096"
# args.n_layer = 32
# args.n_embd = 2560
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"
# args.n_layer = 32
# args.n_embd = 4096

prompt = 'User: Evaluate $(1+2i)6-3i$.\n\nAssistant: <think'
BATCH_SIZE = 320
GENERATION_LENGTH = 4000
BATCH_SIZE = 64
GENERATION_LENGTH = 1000
# BATCH_SIZE = 4
# GENERATION_LENGTH = 100

# we use simple sampling = greedy(logits + noise)
DECODE_NOISE = 1.0
DECODE_TEMP = 0.5

########################################################################################################

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

# init model
print('loading...', args.MODEL_NAME)
model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

# init state
state = [None for _ in range(args.n_layer * 3)]
for i in range(args.n_layer):
    state[i*3+0] = torch.zeros((BATCH_SIZE, args.n_embd), dtype=torch.half, requires_grad=False, device="cuda")
    state[i*3+1] = torch.zeros((BATCH_SIZE, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
    state[i*3+2] = torch.zeros((BATCH_SIZE, args.n_embd), dtype=torch.half, requires_grad=False, device="cuda")

tokens = [tokenizer.encode(prompt) for _ in range(BATCH_SIZE)]
out, state = model.forward_batch(tokens, state)

all_out = []

print('rollout...', end='')

for i in range(GENERATION_LENGTH):
    token = sampler_simple_batch(out, noise=DECODE_NOISE, temp=DECODE_TEMP).tolist()
    all_out.append(token)
    out, state = model.forward_batch(token, state)
    if i % 10 == 0:
        print(i, end=' ', flush=True)
print('\n' + '#'*80 + '\n')

all_out = np.transpose(np.array(all_out), axes=(1,0,2)).squeeze(-1)

for n in range(BATCH_SIZE):
    tokens = all_out[n]
    
    eod = np.flatnonzero(tokens == 0)
    if eod.size:
        tokens = tokens[:eod[0]] # get tokens before eod (token 0)

    out_str = tokenizer.decode(tokens)
    print(out_str)
    print('\n' + '#'*80 + '\n')
