# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'nanogpt-bitnet'
wandb_run_name = 'nanogpt-bitnet-with-gqa-opt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 96
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_kv_heads = 4
n_embd = 448
dropout = 0.2

learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1.5e-5  # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
compile = False # do not torch compile the model
device="mps"