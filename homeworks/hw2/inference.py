import yaml
import torch
import argparse
from torch.nn import functional as F
from transformers import GPT2Tokenizer, set_seed
from libs.model import GPT, GPTConfig
from libs.utils import remove_prefix_from_state_dict    

def inference(prompt, model, tokenizer, num_return_sequences=1, max_length=30, verbose=False, seed=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    model.eval()

    tokens = tokenizer.encode(prompt)
    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Input tokens: {tokens}")
        print('=' * 50)

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    tokens = tokens.repeat(num_return_sequences, 1)
    x = tokens.to(device) # B, T

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)["logits"] # B, T, vocab_size

            # import code; code.interact(local=locals())

            last_token_logits = logits[:, -1, :] # we only care about the last token
            probs = F.softmax(last_token_logits, dim=-1)
            
            # do top-k sampling of 50 (hugging face default). We use top-k sampling so that we can sample from a smaller subset of the possible tokens, which can help with the quality of the generated text.
            topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)

            # select a token from the top-k probabilities
            selected_indices = torch.multinomial(topk_probs, 1) # sample from the top-k probabilities based on the multinomial distribution (instead of argmax or random sampling)
            selected_tokens = topk_indices.gather(dim=-1, index=selected_indices) # out[i] = input[i][index[i]]

            # append the selected token to the input
            x = torch.cat((x, selected_tokens), dim=1)


    # print the generated text
    output = []
    for i in range(num_return_sequences):
        text = tokenizer.decode(x[i].tolist())
        if verbose:
            print(f"> {text}")
        output.append(text)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="the path to the model config")
    parser.add_argument("--ckpt_path", type=str, default=None, help="the path to the checkpoint")
    args = parser.parse_args()

    num_return_sequences = 5
    max_length = 30
    prompt = "Once upon a time"

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT.from_pretrained("gpt2")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_type = config["model"]
    print(f"inferencing with model {model_type}")
    ckpt_path = args.ckpt_path

    gpt_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
    }[config["model"]]
    model = GPT(gpt_config, use_FLASH=config["flash"])

    # since model is compiled, we need to load the state dict into the model
    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        new_state_dict = remove_prefix_from_state_dict(torch.load(ckpt_path), prefix="_orig_mod.")
        model.load_state_dict(new_state_dict)

    output = inference(prompt, model, tokenizer, num_return_sequences=num_return_sequences, max_length=max_length, verbose=True, seed=42)