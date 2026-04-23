import json
import filetype
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count


DATA_ROOT = Path(
    # "/nas_train/app.e0031982/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data"
    "/nas_train/app.e0031982/datasets/FineVision"
)
OUTPUT_ROOT = Path(
    # "/nas_train/app.e0016372/datasets/LLaVA-OneVision-1.5-Instruct-Data"
    "/nas_train/app.e0016372/datasets/FineVision"
)

IMAGE_ROOT = OUTPUT_ROOT / "images"
JSONL_ROOT = OUTPUT_ROOT / "jsonl"

IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
JSONL_ROOT.mkdir(parents=True, exist_ok=True)


def validate_messages(messages):
    if not messages:
        raise ValueError("messages empty after filtering system messages")
    if messages[0]["role"] != "user":
        raise ValueError("conversation must start with user")
    for i in range(1, len(messages)):
        if messages[i - 1]["role"] == messages[i]["role"]:
            raise ValueError(
                f"roles not alternating at index {i}: "
                f"{messages[i - 1]['role']} -> {messages[i]['role']}"
            )


def ensure_image_prefix(messages):
    first_content = messages[0]["content"]
    if not first_content.startswith("<image>"):
        messages[0]["content"] = f"<image>\n{first_content}"


def normalize_messages(raw_messages, has_image=False):
    messages = []
    for idx, item in enumerate(raw_messages):
        if role := item.get("from"):
            if role == "system":
                continue
            role_map = {
                "human": "user",
                "gpt": "assistant",
            }
            if role not in role_map:
                raise ValueError(f"message[{idx}] invalid from: {role}")
            role = role_map[role]
            content = item.get("value")
        elif role := item.get("role"):
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(f"message[{idx}] invalid role: {role}")
            content = item.get("content")
        else:
            assert item.get("user")
            messages.append({"role": "user", "content": item["user"]})
            role, content = "assistant", item["assistant"]
        if not content:
            raise ValueError(f"message[{idx}] empty content")
        messages.append({
            "role": role,
            "content": content,
        })
    validate_messages(messages)
    if has_image:
        ensure_image_prefix(messages)
    return messages


def save_image(image_data, image_dir, sample_name):
    if isinstance(image_data, dict):
        image_bytes = image_data["bytes"]
    else:
        image_bytes = image_data[0]["bytes"]
    suffix = filetype.guess_extension(image_bytes) or "png"
    image_path = image_dir / f"{sample_name}.{suffix}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def process_sample(row, dataset_name, parquet_stem, idx, image_dir):
    record = {
        "id": f"{dataset_name}/{parquet_stem}_{idx}",
        "messages": normalize_messages(
            row.messages,
            has_image=bool(row.images),
        ),
    }
    if images := row.images:
        record["images"] = [
            save_image(
                images,
                image_dir,
                f"{parquet_stem}_{idx}",
            )
        ]
    return record


def process_parquet_file(parquet_path):
    dataset_name = parquet_path.parent.name
    image_dir = IMAGE_ROOT / dataset_name
    jsonl_dir = JSONL_ROOT / dataset_name
    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    output_file = jsonl_dir / f"{parquet_path.stem}.jsonl"
    try:
        df = pd.read_parquet(parquet_path, columns=["texts", "images"])
        # df = pd.read_parquet(parquet_path, columns=["conversations", "image"])
        df.columns = ["messages", "images"]
        with output_file.open("w", encoding="utf-8") as fout:
            for idx, row in enumerate(df.itertuples(index=False)):
                try:
                    record = process_sample(
                        row,
                        dataset_name,
                        parquet_path.stem,
                        idx,
                        image_dir,
                    )
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(
                        f"[ERROR] sample failed | "
                        f"{parquet_path} | idx={idx} | {e}"
                    )
    except Exception as e:
        print(f"[ERROR] parquet failed | {parquet_path} | {e}")


def main():
    parquet_files = list(DATA_ROOT.glob("*/*.parquet"))
    # parquet_files = parquet_files[:2]
    print(len(parquet_files))
    if not parquet_files:
        return
    n_workers = min(cpu_count(), len(parquet_files))
    with Pool(processes=n_workers, maxtasksperchild=10) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    process_parquet_file,
                    parquet_files,
                ),
                total=len(parquet_files),
            )
        )


if __name__ == "__main__":
    main()
