import os
import yaml
import torch
import argparse
import numpy as np
from libs.model import GPT, GPTConfig
from libs.utils import remove_prefix_from_state_dict
from transformers import AutoModelForCausalLM, AutoConfig, GPT2Tokenizer, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="the path to the model config")
    parser.add_argument("--publish", action="store_true", help="publish the model to huggingface")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    out_dir = os.path.dirname(args.config)
    log_path = os.path.join(out_dir, 'main.log')
    with open(log_path, "r") as f:
        lines = f.readlines()

    val_loss = np.array([[int(line.split(' ')[1]), float(line.split(' ')[-1][:-1])] for line in lines if 'val' in line ])
    min_step = np.argmin(val_loss[:,1])
    min_val_loss = val_loss[min_step, 1]
    min_val_step = int(val_loss[min_step, 0])

    print(f"Minimum validation loss: {min_val_loss} at step {min_val_step}")

    ckpt_path = os.path.join(out_dir, f"step_{min_val_step}.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint file not found at {ckpt_path}"

    gpt_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
    }[config["model"]]

    model = GPT(gpt_config, use_FLASH=config["flash"])
    state_dict = torch.load(ckpt_path)
    state_dict = remove_prefix_from_state_dict(state_dict, prefix="_orig_mod.")
    model.load_state_dict(state_dict)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # # Save the model to a directory
    # gpt_config.save_pretrained(f'./hf_models/{config["name"]}')
    # model.save_pretrained(f'./hf_models/{config["name"]}', safe_serialization=False)
    # tokenizer.save_pretrained(f'./hf_models/{config["name"]}')

    if args.publish:
        print(f"Pushing model to huggingface")
        model_card = "Morris88826/Mu-Ruei_Tseng_133007868_350M"
        gpt_config.push_to_hub(model_card)
        model.push_to_hub(model_card, safe_serialization=False)
        tokenizer.push_to_hub(model_card)

    # Register the custom model and config type
    AutoConfig.register("custom-gpt2", GPTConfig)
    AutoModelForCausalLM.register(GPTConfig, GPT)

    # Load the custom model using AutoModel and AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(f'./hf_models/{config["name"]}')
    gpt_config = AutoConfig.from_pretrained(f'./hf_models/{config["name"]}')
    model = AutoModelForCausalLM.from_pretrained(f'./hf_models/{config["name"]}')

    print(f"Model saved to ./hf_models/{config['name']}")



