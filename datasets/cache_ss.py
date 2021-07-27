import random

from PIL import Image
import os
import numpy as np
from datasets.selfdet import selective_search
from functools import partial
from multiprocessing import Pool
import tqdm
import argparse

from main import get_datasets, set_dataset_path, get_args_parser


def cache_ss_item(cache_dir, img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    fn = img_path.split('/')[-1].split('.')[0] + '.npy'
    fp = os.path.join(cache_dir, fn)

    if not os.path.exists(fp):
        boxes = selective_search(img, h, w, res_size=None)
        with open(fp, 'wb') as f:
            np.save(f, boxes)

def main():

    parser = argparse.ArgumentParser('cache ss boxes', add_help=False, parents=[get_args_parser()])
    parser.add_argument('--part', type=int)
    parser.add_argument('--num_m', type=int)
    parser.add_argument('--num_p', type=int)
    parser.add_argument('--shuffle', type=int, default=0)
    args = parser.parse_args()
    set_dataset_path(args)
    args.cache_path = None

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(ROOT_DIR, 'cache')

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet':
        cache_dir = os.path.join(cache_dir, 'ilsvrc', 'ss_box_cache')
    else:
        cache_dir = os.path.join(cache_dir, args.dataset, 'ss_box_cache')

    os.makedirs(cache_dir, exist_ok=True)
    dataset, _ = get_datasets(args)
    files = dataset.files
    if args.shuffle:
        random.shuffle(files)
    chunk_size = int((len(files)) / args.num_m + 1)
    files = files[args.part * chunk_size: (args.part + 1) * chunk_size]
    with Pool(args.num_p) as p:
        r = list(tqdm.tqdm(p.imap(partial(cache_ss_item, cache_dir), files), total=len(files)))


def extract_fns(root):
    files = []
    for (troot, _, files) in os.walk(root, followlinks=True):
        for f in files:
            if f.split('.')[-1].lower() in ['jpeg', 'png']:
                path = os.path.join(troot, f)
                files.append(path)
            else:
                continue
    return files


if __name__ == "__main__":
    main()



