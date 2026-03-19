from pathlib import Path
import json
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm


DATA_ROOT = Path('/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data')
OUT_ROOT = Path('/nas_train/app.e0016372/datasets/LLaVA-OneVision-1.5-Instruct-Data')

IMG_ROOT = OUT_ROOT / 'images'
FILE_ROOT = OUT_ROOT / 'jsonl'

IMG_ROOT.mkdir(parents=True, exist_ok=True)
FILE_ROOT.mkdir(parents=True, exist_ok=True)


def convert_messages(convs):
    messages = []

    for i, m in enumerate(convs):
        role = None
        content = None

        # ---------- 解析 ----------
        if 'from' in m and m['from']:
            if m['from'] == 'human':
                role = 'user'
            elif m['from'] == 'gpt':
                role = 'assistant'
            elif m['from'] == 'system':
                continue
            else:
                raise ValueError(f'item {i} invalid from: {m["from"]}')
            content = m.get('value')

        else:
            role = m.get('role')

            if role == 'system':
                continue

            if role not in ['user', 'assistant']:
                raise ValueError(f'item {i} invalid role: {role}')

            content = m.get('content')

        # ---------- content 校验 ----------
        if not content:
            raise ValueError(f'item {i} content is empty')

        messages.append({
            'role': role,
            'content': content
        })

    # ---------- 整体校验 ----------
    if not messages:
        raise ValueError('messages is empty after filtering system')

    # 必须以 user 开头
    if messages[0]['role'] != 'user':
        raise ValueError('conversation must start with user')

    # 必须交替
    for i in range(1, len(messages)):
        prev = messages[i - 1]['role']
        curr = messages[i]['role']

        if prev == curr:
            raise ValueError(
                f'role not alternating at index {i}: {prev} -> {curr}'
            )

    return messages


def process_parquet(parquet_path):
    dataset = parquet_path.parent.name

    img_dir = IMG_ROOT / dataset
    file_dir = FILE_ROOT / dataset

    img_dir.mkdir(parents=True, exist_ok=True)
    file_dir.mkdir(parents=True, exist_ok=True)

    out_file = file_dir / f'{parquet_path.stem}.jsonl'

    try:
        df = pd.read_parquet(parquet_path, columns=['conversations', 'image'])

        with out_file.open('w', encoding='utf-8') as fout:
            for idx, row in enumerate(df.itertuples(index=False)):
                try:
                    uid = f'{dataset}/{parquet_path.stem}_{idx}'

                    messages = convert_messages(row.conversations)
                    data = {'id': uid, 'messages': messages}

                    if img_data := row.image:
                        img = Image.open(BytesIO(img_data['bytes']))
                        ext = 'jpg' if img.format in ('JPEG', 'JPG') else 'png'
                        fnm = f'{parquet_path.stem}_{idx}.{ext}'
                        img.save(img_dir / fnm)
                        data['images'] = [f'{dataset}/{fnm}']

                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f'[ERROR] sample failed | {parquet_path} | idx={idx} | {e}')
    except Exception as e:
        print(f'[ERROR] parquet failed | {parquet_path} | {e}')


def main():
    parquet_files = list(DATA_ROOT.glob('*/*.parquet'))
    print(len(parquet_files))
    # parquet_files = parquet_files[:10]
    n_jobs = min(cpu_count(), len(parquet_files))

    with Pool(processes=n_jobs, maxtasksperchild=10) as pool:
        list(tqdm(pool.imap_unordered(process_parquet, parquet_files), total=len(parquet_files)))

    # list(Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
    #     delayed(process_parquet)(parquet_path)
    #     for parquet_path in tqdm(parquet_files)
    # ))


if __name__ == '__main__':
    main()
