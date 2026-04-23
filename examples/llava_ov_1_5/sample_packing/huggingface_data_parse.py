from datasets import load_dataset
from multiprocessing import Pool
from tool import cfg, get_init_file
import os
from functools import partial
from pathlib import Path
from tqdm import tqdm
import hashlib
import shutil
import json
import re
import numpy as np
from PIL import Image
import io


def check_caption(content):
    if content.lower().startswith(("i'm sorry", "i am sorry", "i cannot", "i can't")):
        return False

    words = re.findall(r'\b\w+\b', content.lower())
    if len(words) >= 8:
        for i in range(len(words) - 7):
            if len(set(words[i:i + 8])) == 1:
                return False

    if len(content) > 3500 or len(content) < 50:
        return False

    return True


def check_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        if not image_data:
            return False

        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)

        if np.all(img_array == 0):
            return False

        return True

    except Exception as e:
        return False


def parse_item(item, dst_dir):
    try:
        if 'id' in item:
            name = item['id'].replace('/', '_')
            name = os.path.splitext(name)[0]
        else:
            raw = item['caption']
            name = hashlib.md5(raw.encode('utf-8')).hexdigest()
            
        image_path = os.path.join(dst_dir, name + '.jpg')
        image = item['image'].convert('RGB')
        image.save(
            image_path,
            quality=95,
            subsampling=0,
            optimize=True,
            progressive=False
        )

        if cfg['data']['filter_with_caption'] and not check_caption(item['caption']):
            return

        if cfg['data']['filter_with_image'] and not check_image(image_path):
            return

        json_data = {
            'messages': [
                {'role': 'user', 'content': '<image>'},
                {'role': 'assistant', 'content': item['caption']}
            ],
            'images': [image_path]
        }

        json_path = os.path.join(dst_dir, name + '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)

    except Exception as e:
        print(f"{name} has exception: {e}")


def process_parquet(parquet_path, dst_dir):
    try:
        dataset = load_dataset(
            'parquet',
            data_files=str(parquet_path),
            split='train'
        )

        for item in dataset:
            parse_item(item, dst_dir)

    except Exception as e:
        print(f'Error processing {parquet_path}: {e}')


def main():
    # data_root = Path(cfg['hf_data'])
    # dst_dir = get_init_file()[-1]
    data_root = Path('/nas_train/app.e0016372/datasets/OmniScience/data')
    dst_dir = Path('/nas_train/app.e0016372/datasets/OmniScience/images')
    shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_root.glob('**/*.parquet'))
    workers = min(len(parquet_files), os.cpu_count())

    print(f'Total parquet files: {len(parquet_files)}')

    with Pool(processes=workers) as pool:
        list(tqdm(
            pool.imap_unordered(
                partial(process_parquet, dst_dir=dst_dir),
                parquet_files
            ),
            total=len(parquet_files),
            desc='Processing parquet files'
        ))


if __name__ == '__main__':
    main()
