import yaml
import argparse
from libs.model import GPT, GPTConfig
from libs.utils import remove_prefix_from_state_dict
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="the path to the model config")
    parser.add_argument("-c", "--ckpt_path", type=str, default=None, help="the path to the checkpoint")
    args = parser.parse_args()

    AutoModel.register(GPTConfig, GPT)
    AutoModelForCausalLM.register(GPTConfig, GPT)

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    gpt_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, norm_method=config["norm_method"], act_method=config["act_method"], RoPE=config["use_RoPE"], group_size=config["group_size"]),
    }[config["model"]]

    model = GPT(gpt_config, use_FLASH=config["flash"])
    state_dict = model.state_dict()
    state_dict = remove_prefix_from_state_dict(state_dict, prefix="module._orig_mod.")
    model.load_state_dict(state_dict)

    # Save the model to a directory
    gpt_config.save_pretrained('./hf_models/custom-gpt2')
    model.save_pretrained('./hf_models/custom-gpt2', safe_serialization=False)


    # Register the custom model and config type
    AutoConfig.register("custom-gpt2", GPTConfig)
    AutoModel.register(GPTConfig, GPT)

    # Load the custom model using AutoModel and AutoConfig
    config = AutoConfig.from_pretrained("./hf_models/custom-gpt2")
    model = AutoModel.from_pretrained("./hf_models/custom-gpt2")
