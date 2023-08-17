# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)



model: HookedTransformer = HookedTransformer.from_pretrained("gelu-1l")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
data = load_dataset("NeelNanda/c4-code-20k", split="train")
c4_data = load_dataset("NeelNanda/c4-10k", split="train")
code_data = load_dataset("NeelNanda/code-10k", split="train")
# %%
num_prompts = 200
text = [i for i in c4_data[:num_prompts]["text"] if len(i)>2000]
len(text)
# %%
def capitalize_string(s):
    return " ".join([x.capitalize() for x in s.split(" ")])
cap_text = [capitalize_string(x) for x in text]
tokens = model.to_tokens(text)[:, :256]
cap_tokens = model.to_tokens(cap_text)[:, :256]
# %%
normal_logits, normal_cache = model.run_with_cache(tokens)
normal_ave_neuron_acts = normal_cache["post", 0].mean([0, 1])
cap_logits, cap_cache = model.run_with_cache(cap_tokens)
cap_ave_neuron_acts = cap_cache["post", 0].mean([0, 1])

scatter(x=normal_ave_neuron_acts, y=cap_ave_neuron_acts, hover=np.arange(d_mlp), xaxis="Normal", yaxis="Capitalized", title="Average Neuron Activation")
# %%
ni = 424
win = model.W_in[0, :, ni]
wout = model.W_out[0, ni, :]
cap_stack, resid_labels = cap_cache.get_full_resid_decomposition(0, True, False, True, return_labels=True)
normal_stack, resid_labels = normal_cache.get_full_resid_decomposition(0, True, False, True, return_labels=True)
line([(cap_stack @ win).mean([1, 2]), (normal_stack @ win).mean([1, 2])], x=resid_labels, title="Ave contribution to neuron", line_labels=["capitalized", "normal"])
line([(cap_stack @ win).std([1, 2]), (normal_stack @ win).std([1, 2])], x=resid_labels, title="Ave std of contribution to neuron", line_labels=["capitalized", "normal"])
# %%
head_dna = model.W_E @ model.W_V[0] @ model.W_O[0] @ win
for i in range(8):
    nutils.show_df(nutils.create_vocab_df(head_dna[i], model=model).head(100))
# %%
full_vocab = model.to_str_tokens(np.arange(d_vocab))
has_space = np.array([(i[0]==" " if len(i)>0 else False) for i in full_vocab])
begins_cap = np.array([(i.strip()[0]!=i.strip()[0].lower()  if len(i.strip())>0 else False) for i in full_vocab])
is_word = np.array([i.isalpha() or (i[0]==" " and i[1:].isalpha()) if len(i)>0 else False for i in full_vocab])
is_capital = has_space & begins_cap & is_word
is_not_capital = has_space & (~begins_cap) & is_word
is_middle = (~has_space) & (~begins_cap) & is_word
vocab_df = pd.DataFrame(dict(
    i=np.arange(d_vocab),
    tok=nutils.process_tokens(full_vocab, model),
    s=full_vocab,
    has_space=has_space,
    begins_cap=begins_cap,
    is_capital=is_capital,
    is_not_capital=is_not_capital,
    is_middle=is_middle,
    is_word=is_word,
))
vocab_df.iloc[1000:1009]
# %%
overall_dna = torch.cat([head_dna, (model.W_E @ win)[None]])
line([overall_dna[:, is_capital].mean(-1), overall_dna[:, ~is_capital].mean(-1), overall_dna[:, has_space & (~begins_cap) & is_word].mean(-1), overall_dna[:, is_middle].mean(-1)], title="Grouped DNA by head", line_labels=["is_capital", "not is_capital", "is_not_capital", "is_middle"], x=model.all_head_labels()+["W_E"])
# %%
is_not_capital_t = torch.tensor(is_not_capital).cuda()
is_not_capital_token = is_not_capital_t[tokens]

is_capital_t = torch.tensor(is_capital).cuda()
is_capital_token = is_capital_t[tokens]

neuron_acts_pre = normal_cache["pre", 0][:, :, ni]
neuron_acts = normal_cache["post", 0][:, :, ni]

print((neuron_acts * is_not_capital_token).sum() / is_not_capital_token.sum())
print((neuron_acts * is_capital_token).sum() / is_capital_token.sum())
# %%
not_capital_acts = neuron_acts[is_not_capital_token]
capital_acts = neuron_acts[is_capital_token]
histogram(not_capital_acts)
histogram(capital_acts)
# %%
category = []
for i in range(d_vocab):
    if is_capital[i]:
        category.append("capital")
    elif is_not_capital[i]:
        category.append("not capital")
    elif is_middle[i]:
        category.append("middle")
    elif not is_word[i]:
        category.append("not word")
    else:
        category.append("other")
vocab_df["category"] = category
vocab_df
# %%
token_df = copy.deepcopy(vocab_df.iloc[tokens.flatten().tolist()])
token_df["batch"] = [b for b in range(tokens.shape[0]) for p in range(tokens.shape[1])]
token_df["pos"] = [p for b in range(tokens.shape[0]) for p in range(tokens.shape[1])]
token_df["pre"] = neuron_acts_pre.flatten().tolist()
token_df["post"] = neuron_acts.flatten().tolist()

# %%
is_start = []
FULL_STOP = model.to_single_token(".")
QUESTION = model.to_single_token("?")
EXCL = model.to_single_token("!")
for b in range(tokens.shape[0]):
    is_start.append(False)
    for p in range(1, tokens.shape[1]):
        if (tokens[b, p-1] == FULL_STOP) or (tokens[b, p-1] == EXCL) or (tokens[b, p-1] == QUESTION):
            is_start.append(True)
        else:
            is_start.append(False)
token_df["is_start"] = is_start
token_category = token_df["category"].values
token_df["category_2"] = [token_category[i] if not is_start[i] else "start" for i in range(len(token_df))]

px.histogram(token_df, x="post", color="category_2", barmode="overlay", histnorm="percent").show()
px.histogram(token_df.query("category=='capital'"), x="post", color="is_start", barmode="overlay", histnorm="percent", title="Neuron acts given capitalized word", marginal="box")

# %%
nutils.show_df(token_df.query("category=='capital' & ~is_start & post < 0.").head(100).reset_index())

# %%
b = 63
nutils.create_html(model.to_str_tokens(tokens[b]), neuron_acts[b])
# %%
print(set(token_df.query("category=='capital' & ~is_start & post < 0.").tok.to_list()))
# %%
(token_df.query("category=='capital' & ~is_start").groupby("tok")["post"].max() - token_df.query("category=='capital' & ~is_start").groupby("tok")["post"].min()).sort_values(ascending=False)
# %%
token_df[token_df.s==" You"]
# %%
import circuitsvis as cv
cv.attention.attention_patterns(tokens=model.to_str_tokens(tokens[63]), attention=normal_cache["pattern", 0][63])
# %%
patterns = normal_cache["pattern", 0]
curr_token = patterns.diagonal(0, -2, -1).mean([0, 2])
prev_token = patterns.diagonal(-1, -2, -1).mean([0, 2])
prev_2_token = patterns.diagonal(-2, -2, -1).mean([0, 2])
bos_token = patterns[:, :, :, 0].mean([0, 2])
line([curr_token, prev_token, prev_2_token, bos_token], x=model.all_head_labels(), title="Average attention pattern", line_labels=["curr", "prev", "prev_2", "bos"])
curr_token = patterns.diagonal(0, -2, -1).std([0, 2])
prev_token = patterns.diagonal(-1, -2, -1).std([0, 2])
prev_2_token = patterns.diagonal(-2, -2, -1).std([0, 2])
bos_token = patterns[:, :, :, 0].std([0, 2])
line([curr_token, prev_token, prev_2_token, bos_token], x=model.all_head_labels(), title="Std attention pattern", line_labels=["curr", "prev", "prev_2", "bos"])
# %%
normalised_W_E = model.W_E[None, :, :] + model.W_pos[100:110, None, :]
normalised_W_E = normalised_W_E / normalised_W_E.std(dim=-1, keepdim=True)
normalised_W_E = normalised_W_E.mean(0)
normalised_W_E.shape
begins_space_embed = normalised_W_E[is_word & has_space].mean(0)
not_begins_space_embed = normalised_W_E[is_word & (~has_space)].mean(0)
is_capital_embed = normalised_W_E[is_capital].mean(0)
is_not_capital_embed = normalised_W_E[is_not_capital].mean(0)
full_stop_embed = normalised_W_E[model.to_single_token(".")]
embeds = torch.stack([begins_space_embed, not_begins_space_embed, is_capital_embed, is_not_capital_embed, full_stop_embed])
embeds -= normalised_W_E.mean(0)
embed_labels = ["begins_space", "not_begins_space", "is_capital", "is_not_capital", "full_stop"]
imshow(embeds @ model.W_K[0] @ model.W_Q[0].transpose(-2, -1) @ embeds.T, x=embed_labels, y=embed_labels, facet_col=0, facet_labels=model.all_head_labels(), xaxis="Query", yaxis="Key", title="Average QK circuit for heads")

line(torch.cat([(embeds @ model.W_V[0] @ model.W_O[0] @ win).T, (embeds @ win)[:, None]], dim=1), x=model.all_head_labels() + ["W_E"], line_labels=embed_labels, title="Average OV Circuit for Heads")

# %%
begins_space_unembed = model.W_U.T[is_word & has_space].mean(0)
not_begins_space_unembed = model.W_U.T[is_word & (~has_space)].mean(0)
is_capital_unembed = model.W_U.T[is_capital].mean(0)
is_not_capital_unembed = model.W_U.T[is_not_capital].mean(0)
full_stop_unembed = model.W_U.T[model.to_single_token(".")]
unembeds = torch.stack([begins_space_unembed, not_begins_space_unembed, is_capital_unembed, is_not_capital_unembed, full_stop_unembed])
unembeds -= model.W_U.T.mean(0)
unembed_labels = ["begins_space", "not_begins_space", "is_capital", "is_not_capital", "full_stop"]
line(unembeds @ wout, x=unembed_labels, title="DLA of neuron")


px.histogram(x=to_numpy(wout@model.W_U), color=category, histnorm="percent", barmode="overlay", marginal="box", title="Direct Logit Attr of neuron", hover_name=full_vocab)
# %%
px.histogram(token_df.query("is_middle"), x="post", barmode="overlay", histnorm="percent", title="Neuron acts given capitalized word")

is_capital_shift = np.zeros(len(token_df), dtype=bool)
is_capital_shift[1:] = token_df.is_capital.values[:-1]
token_df['prev_is_capital'] = is_capital_shift
is_capital_shift_2 = np.zeros(len(token_df), dtype=bool)
is_capital_shift_2[2:] = token_df.is_capital.values[:-2]
is_not_capital_shift = np.zeros(len(token_df), dtype=bool)
is_not_capital_shift[1:] = token_df.is_not_capital.values[:-1]
token_df['two_prev_is_capital'] = is_capital_shift_2 & ~is_not_capital_shift
token_df["is_middle_cap_word"] = token_df.prev_is_capital | token_df.two_prev_is_capital
token_df["label"]=[f"B{b}/P{p}" for b, p in zip(token_df.batch, token_df.pos)]
px.histogram(token_df.query("is_middle"), x="post", barmode="overlay", 
             color="prev_is_capital", histnorm="percent", title="Neuron acts for mid-word tokens given prev_is_capital", hover_name="label", marginal="box")
# %%
