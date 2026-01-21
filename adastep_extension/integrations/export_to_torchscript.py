"""Export AdaStep + ACT policy to TorchScript (edge-friendly). Template for CI smoke tests."""
import argparse
import torch
import numpy as np


def make_dummy_model():
    # tiny model for CI smoke
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64*64*3, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))


def export(out_path: str, device: str = 'cpu'):
    m = make_dummy_model().to(device).eval()
    example = torch.randn(1, 3, 64, 64).to(device)
    ts = torch.jit.trace(m, example)
    ts.save(out_path)
    print('Saved TorchScript to', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='adastep_smoke.pt')
    args = p.parse_args()
    export(args.out)
