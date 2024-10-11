import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.config)

    log_path = os.path.join(out_dir, 'main.log')
    with open(log_path, "r") as f:
        lines = f.readlines()

    train_loss = np.array([[int(line.split(' ')[1]), float(line.split(' ')[-1][:-1])] for line in lines if 'train' in line ])
    val_loss = np.array([[int(line.split(' ')[1]), float(line.split(' ')[-1][:-1])] for line in lines if 'val' in line ])

    model_version = os.path.basename(out_dir)
    plt.plot(train_loss[:,0], train_loss[:,1], label='train')
    plt.plot(val_loss[:,0], val_loss[:,1], label='val')
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'{model_version} Loss')
    plt.savefig(os.path.join(out_dir, 'loss.png'))