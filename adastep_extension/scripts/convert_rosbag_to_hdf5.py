#!/usr/bin/env python3
"""rosbag -> hdf5 converter (lightweight, from CSDN tutorial).
Usage: python scripts/convert_rosbag_to_hdf5.py --bag <path> --out <out.hdf5>
"""
import argparse
import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm


def convert_rosbag_stub(bag_path, save_path, image_topics, joint_topics):
    # This is a simplified converter: it expects that image topics and joint topics
    # are already synchronized in the rosbag. For robust extraction use rosbag API.
    print("Converter (stub) — this script provides a runnable template.")
    print(f"Input bag: {bag_path}\nOutput hdf5: {save_path}")
    # === placeholder: in real use open rosbag and extract messages ===
    # Here we create a tiny synthetic example to validate downstream code.
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    frames = np.stack([img for _ in range(10)], axis=0)
    joints = np.zeros((10, 14), dtype=np.float32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as h5f:
        obs = h5f.create_group('observations')
        imgs = obs.create_group('images')
        imgs.create_dataset('top', data=frames)
        obs.create_dataset('qpos', data=joints)
        obs.create_dataset('qvel', data=np.zeros_like(joints))
        h5f.create_dataset('action', data=joints)

    print('Wrote synthetic HDF5 sample (for smoke test).')


def main():
    p = argparse.ArgumentParser(description='rosbag -> hdf5 (template)')
    p.add_argument('--bag', type=str, help='input rosbag (optional for smoke)')
    p.add_argument('--out', type=str, required=True, help='output .hdf5')
    p.add_argument('--image-topics', nargs='+', default=['/camera_f/color/compressed','/camera_l/color/compressed','/camera_r/color/compressed'])
    p.add_argument('--joint-topics', nargs='+', default=['/puppet/joint_left','/puppet/joint_right'])
    args = p.parse_args()

    convert_rosbag_stub(args.bag, args.out, args.image_topics, args.joint_topics)


if __name__ == '__main__':
    main()
