import os
def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)

def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict