#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 使用方式
python s1_get_tokenlens_v4-sft.py --config ./configs/s1_config_MR_sft_780k.yaml
"""

import os
import json
import orjson
import threading
import logging
import psutil
import tempfile
import queue
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import merge
from PIL import Image
from jinja2 import Template
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from qwen_vl_utils import fetch_image
from queue import Empty
import multiprocessing
from multiprocessing import Pool, Manager, Value

# 声明全局的跨进程计数器（在主模块中定义，让子进程继承）
global_total_counter = None

# ✅ 解析命令行参数
parser = argparse.ArgumentParser(description="Token Length Processor")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
parser.add_argument("--log-level", type=str, default=None,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Override log level from config")
args = parser.parse_args()

# ✅ 加载配置文件
CONFIG_PATH = Path(__file__).parent.joinpath('configs/s1_config_BMR_sft_780k.yaml')
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# ✅ 从配置中读取参数，覆盖原有常量
MAX_TOKEN_LEN = cfg['sample']['max_len']
task_type = cfg['sample']['task_type']
DEL_ONE_TOKEN = cfg['sample']['del_one_token']

DEFAULT_DIRECTORY = Path(cfg['data']['directory'])
OUTPUT_DIR = DEFAULT_DIRECTORY.parent.joinpath('output_dir')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR.joinpath(cfg['data']['output_base'])
TOKEN_INFO_FILE = OUTPUT_DIR.joinpath(cfg['data']['output_token'])
CKPT_DIR = cfg['model']['checkpoint']
MIN_PIXELS = cfg['image']['min_pixels']
MAX_PIXELS = cfg['image']['max_pixels']
image_resolution = cfg['image']['baidu_resolution']
TIME_OUT = cfg['processing']['time_out']
# 归并参数（仅两级：stage0 → stage1）
STAGE1_CHUNK = cfg['processing']['stage1_merge_chunk']
chunk_size = cfg['processing']['chunk_size']
n_workers = cfg['processing']['n_workers']
MIN_WORKERS = cfg['processing']['min_workers']
MAX_WORKERS = cfg['processing']['max_workers']
use_shm = cfg['logging']['use_shm']
log_level = cfg['logging']['level']
log_file = OUTPUT_DIR.joinpath(cfg['logging']['file'])
if args.log_level:
    log_level = args.log_level.upper()

# 日志配置 - 详细记录数据流向和合并过程
file_handler = logging.FileHandler(
    log_file,
    delay=True,
    encoding='utf-8'
)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

EXTENSIONS = (".json", ".jpg")


temp_dir = '/dev/shm' if use_shm else None  # None 表示使用系统默认临时目录

def count_lines(file_path):
    """统计文件有效行数（非空且含分隔符）"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip() and ':' in line.strip())
    except Exception as e:
        logger.error(f"❌ 统计文件 {file_path} 行数失败: {str(e)}")
        return 0

def find_paired_files(directory):
    directory = Path(directory)
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith('.json')}
    img_set  = {f[:-4] for f in files if f.lower().endswith(('.jpg', '.jpeg'))}
    paired = json_set & img_set
    logger.info(f"找到 {len(paired)} 对匹配文件")
    return paired

def find_valid_files(fname_json, rel_img_path):
    from s1_mr_sft_data_proc_indcoding import split_json_file
    valid_names = split_json_file(
                    fname_json, 
                    rel_img_path,
                    chunk_dim=2000,
                    m=8
    )
    return valid_names

def find_valid_json(directory):
    directory = Path(directory)
    files = os.listdir(directory)
    json_set = {f[:-5] for f in files if f.lower().endswith('.json')}
    logger.info(f"找到 {len(json_set)} 个json文件")
    return json_set    

def write_base_names_to_file(base_names, output_file):
    """将配对文件名写入文件"""
    try:
        content = "\n".join(sorted(base_names)) + "\n"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"ℹ️ 已将 {len(base_names)} 个配对文件名写入 {output_file}")
    except Exception as e:
        logger.error(f"❌ 写入 {output_file} 失败: {str(e)}")
        raise


def read_lines_in_chunks(file_path, chunk_size):
    """按块读取文件内容"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} 不存在")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = [line.strip() for _, line in zip(range(chunk_size), f) if line.strip()]
            if not chunk:
                break
            logger.info(f"ℹ️ 读取数据块，包含 {len(chunk)} 个样本")
            yield chunk


# 预编译模板
"""
Todo:
    1) 放到 .yaml 中
    2) 加入非 ”jinja2+processor“ 支持（用户自定义处理函数） 
"""
if task_type=="pretrain":
    CAP_TEMPLATE = Template("<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>")
elif task_type=="sft":
    chat_template  = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
    CAP_TEMPLATE = Template(chat_template)
    pass

def process_sample(json_path, img_path, processor):
    """处理单个样本，返回(token_len, 文件名)"""
    try:
        if not Path(json_path).exists():
            raise FileNotFoundError(f"❌ JSON文件不存在: {json_path}")
        # if not Path(img_path).exists():
        #     raise FileNotFoundError(f"❌ 图片文件不存在: {img_path}")

        # 读取并渲染JSON内容
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # with open(json_path, 'rb') as f:
        #     json_data = orjson.loads(f.read())
        if task_type=="pretrain":
            txt_input = CAP_TEMPLATE.render(captions=json_data['captions'])
        elif task_type=="sft":
            # txt_input = CAP_TEMPLATE.render(json_data)
            txt_input = CAP_TEMPLATE.render(json_data,tokenize=False, add_generation_prompt=False)
        if img_path=="_____.jpg":
            img_input = None
        else:
            def baidu_img_proc(image, image_resolution):
                image = Image.open(image)
                if max(image.width, image.height) > image_resolution:
                    resize_factor = image_resolution / max(image.width, image.height)
                    width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                    image = image.resize((width, height), resample=Image.NEAREST)

                return image

            # if image_resolution:
            #     img_path = baidu_img_proc(img_path, image_resolution)
                
            
            img_input = fetch_image({
                'type': 'image',
                'image': img_path,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            })
        # print(img_input)
        # 计算token长度
        base_name = Path(json_path).stem
        inputs = processor(
            text=[txt_input],
            images=img_input,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        # print(inputs["input_ids"])
        # print(inputs["input_ids"].shape)
        return (inputs["input_ids"].shape[1], base_name)

    except Exception as e:
        return (None, f"❌ 处理失败 [{Path(json_path).stem}]: {str(e)}")


def get_adaptive_workers(min_workers=20, max_workers=96):
    """根据系统负载调整线程数"""
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem_usage = psutil.virtual_memory().percent
        if cpu_usage > 80 or mem_usage > 85:
            adjusted = max(min_workers, max_workers // 2)
            logger.info(f"系统负载过高，线程数调整为 {adjusted} (CPU: {cpu_usage}%, 内存: {mem_usage}%)")
            return adjusted
        return max_workers
    except Exception as e:
        logger.warning(f"获取系统负载失败，使用默认线程数 {max_workers}: {str(e)}")
        return max_workers

gt_maxlen=0
def merge_files_by_token(input_files, output_file, max_token=MAX_TOKEN_LEN):
    """合并多个已排序文件，按token_len升序，过滤掉 > max_token 的数据，返回(输出路径, 数据条数)"""
    if not input_files:
        logger.warning("⚠️ 没有文件可合并")
        return (None, 0)

    # 验证输入文件并统计总数据量
    valid_files = []
    total_lines = 0
    for f in input_files:
        line_count = count_lines(f)
        if line_count > 0:
            valid_files.append(f)
            total_lines += line_count
            logger.debug(f"ℹ️ 待合并文件 {os.path.basename(f)} 包含 {line_count} 条数据")
        else:
            logger.warning(f"⚠️ 文件 {os.path.basename(f)} 为空或无效，跳过")

    if not valid_files:
        return (None, 0)

    # 定义排序键（按token_len整数排序）
    def sort_key(line):
        token_str = line.strip().split(':')[-1]
        return int(token_str)

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # 创建所有文件的迭代器
            iterators = []
            file_handles = []
            for fpath in valid_files:
                try:
                    fh = open(fpath, 'r', encoding='utf-8')
                    file_handles.append(fh)
                    iterators.append(((sort_key(line), line) for line in fh))
                except Exception as e:
                    logger.error(f"❌ 打开文件 {os.path.basename(fpath)} 失败: {str(e)}")

            # # 归并排序并写入
            # for _, line in merge(*iterators, key=lambda x: x[0]):
            #     out_f.write(line)
            # 归并排序并写入，过滤掉 > max_token 的行(后续可添加其他条件)
            filtered_max_len = 0
            for _, line in merge(*iterators, key=lambda x: x[0]):
                _, token_str = line.strip().split(':', 1)
                if int(token_str) <= max_token:   # ← 只保留 ≤ 8192
                    out_f.write(line)
                else:
                    logger.warning(f"⚠️ token长度：{token_str} > {max_token}: 剔除!!")
                    filtered_max_len+=1
                    gt_maxlen

            # 关闭所有文件句柄
            for fh in file_handles:
                try:
                    fh.close()
                except Exception as e:
                    logger.warning(f"⚠️ 关闭文件 {fh.name} 失败: {str(e)}")

        # 验证输出文件数据完整性
        output_lines = count_lines(output_file)+filtered_max_len
        if output_lines != total_lines:   # 过滤掉不满足条件的
            logger.error(f"❌ 合并数据丢失！输入 {total_lines} 条，输出 {output_lines} 条，已删除错误文件")
            if os.path.exists(output_file):
                os.remove(output_file)
            return (None, 0)
        else:
            logger.info(f"✅ 📊 合并成功，输入 {total_lines} 条，输出 {output_lines-filtered_max_len} 条（token ≤ {max_token}）的数据")

        return (output_file, output_lines-filtered_max_len)
    except Exception as e:
        logger.error(f"❌ 合并文件失败: {str(e)}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                logger.warning(f"⚠️ 删除失败文件 {output_file} 失败: {str(e)}")
        return (None, 0)


def stage1_merger(input_queue, chunk_size, stage1_files, stop_event):
    """
    修复版stage1合并线程
    - 确保所有stage0文件被合并，包括最后不足10个的文件
    - 解决线程超时和数据丢失问题
    """
    buffer = []
    batch_counter = 0
    logger.info(f"💡 stage1合并线程启动，每 {chunk_size} 个stage0文件合并一次")

    try:
        # 循环条件：队列有文件 或 缓冲区有文件 或 未收到停止信号
        while (not input_queue.empty()) or buffer or (not stop_event.is_set()):
            # 从队列取文件（带超时防止永久阻塞）
            if not input_queue.empty():
                try:
                    file_path = input_queue.get(timeout=1)  # 超时1秒，避免永久阻塞
                    buffer.append(file_path)
                    input_queue.task_done()
                    logger.debug(f"ℹ️ stage1接收文件 {os.path.basename(file_path)}，当前缓冲区: {len(buffer)}/{chunk_size}")

                    # 达到合并数量则执行合并
                    if len(buffer) >= chunk_size:
                        batch_counter += 1
                        merged_file = tempfile.NamedTemporaryFile(
                            mode='w', delete=False,
                            prefix=f"stage1_batch{batch_counter:03d}_",
                            encoding='utf-8',
                            dir=temp_dir
                        ).name
                        
                        # 执行合并
                        merged_path, line_count = merge_files_by_token(buffer, merged_file)
                        if merged_path and line_count > 0:
                            stage1_files.append(merged_path)
                            logger.info(f"📊 stage1批次 {batch_counter} 完成: {os.path.basename(merged_path)}，包含 {line_count} 条数据（合并了 {len(buffer)} 个文件）")
                        else:
                            logger.warning(f"⚠️ stage1批次 {batch_counter} 合并失败，跳过该批次")

                        # 清空缓冲区
                        buffer = []
                except Empty:
                    continue  # 队列为空时继续循环
                except Exception as e:
                    logger.error(f"❌ stage1处理文件时错误: {str(e)}", exc_info=True)
            else:
                # 队列为空时，检查是否需要强制合并剩余文件
                if buffer and stop_event.is_set():
                    # 收到停止信号且缓冲区有文件，强制合并
                    batch_counter += 1
                    merged_file = tempfile.NamedTemporaryFile(
                        mode='w', delete=False,
                        prefix=f"stage1_remaining_batch{batch_counter:03d}_",
                        encoding='utf-8',
                        dir=temp_dir
                    ).name
                    
                    merged_path, line_count = merge_files_by_token(buffer, merged_file)
                    if merged_path and line_count > 0:
                        stage1_files.append(merged_path)
                        logger.info(f"📊 stage1剩余文件合并完成: {os.path.basename(merged_path)}，包含 {line_count} 条数据（合并了 {len(buffer)} 个文件）")
                    else:
                        logger.warning(f"❌ stage1剩余文件合并失败，数据可能丢失")
                    buffer = []
                else:
                    # 短暂休眠，减少CPU占用
                    threading.Event().wait(0.5)

        # 最终检查：确保缓冲区为空（防止遗漏）
        if buffer:
            logger.error(f"❌ stage1线程退出时缓冲区仍有 {len(buffer)} 个文件未处理！数据丢失")

    except Exception as e:
        logger.error(f"❌ stage1线程异常退出: {str(e)}", exc_info=True)
    finally:
        logger.info(f"📊 stage1线程退出，共生成 {len(stage1_files)} 个文件")

# 新增：每个进程的处理函数（负责处理一个大chunk）
def process_chunk(args):
    """
    单个进程的处理逻辑：处理一个大chunk，内部用多线程并行
    
    Args:
        args: 包含chunk数据、处理器配置、队列等参数的元组
    """
    # 从全局变量获取计数器，而非参数
    global global_total_counter
    
    chunk_idx, chunk, ckpt_dir, min_pixels, max_pixels, stage0_queue = args
    processor = None
    processed_count = 0  # 记录当前进程处理的有效样本数
    
    
    try:
        # 每个进程单独初始化处理器（进程间不能共享processor实例）
        # quant_config = BitsAndBytesConfig(load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(
            ckpt_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
            use_fast=False
        )
        # 生成当前chunk的文件路径列表
        full_paths = []
        for fn in chunk:
            cur_json = str(DEFAULT_DIRECTORY / f"{fn}.json")
            # logger.info(f"👉 进程 {multiprocessing.current_process().name} json文件：{cur_json}.....{type(cur_json)}")
            if f"{fn}.json".startswith("__img--output_"):
                cur_img = "_____.jpg"
                # cur_img = str(DEFAULT_DIRECTORY / f"{cur_img}")
            else:     
                with open(cur_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cur_img = data['images'][0]
                    cur_img = str(DEFAULT_DIRECTORY / f"{cur_img}")
            full_paths.append(cur_json)
            full_paths.append(cur_img)
            # print(f"--------------cur_json:{cur_json}, cur_img:{cur_img}-------------------")
            

        n_samples = len(chunk)
        logger.info(f"👉 进程 {multiprocessing.current_process().name} 开始处理块 {chunk_idx}，包含 {n_samples} 个样本")
        
        # 进程内创建线程池（复用线程）
        n_workers = get_adaptive_workers(min_workers=MIN_WORKERS, max_workers=MAX_WORKERS)  # 单个进程的线程数可适当减少
        chunk_results = []
        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix=f"proc-{multiprocessing.current_process().pid}-thread"
        ) as executor:
            tasks = [
                executor.submit(
                    process_sample,
                    full_paths[idx*2],
                    full_paths[idx*2+1],
                    processor
                ) for idx in range(n_samples)
            ]
            
            # 收集线程任务结果
            for future in as_completed(tasks):
                try:
                    token_len, name = future.result()
                    if DEL_ONE_TOKEN:
                        token_len += 1
                    if token_len is not None:
                        chunk_results.append((token_len, name))
                        processed_count += 1  # 统计有效样本
                    else:
                        logger.warning(name)
                except Exception as e:
                    logger.error(f"❌ 进程内任务错误: {str(e)}")
        
        # 写入stage0文件并放入跨进程队列
        if chunk_results:
            chunk_results_sorted = sorted(chunk_results, key=lambda x: x[0])
            with tempfile.NamedTemporaryFile(
                mode='w+', delete=False,
                prefix=f"stage0_chunk{chunk_idx:03d}_",
                encoding='utf-8',
                dir=temp_dir  
            ) as f:
                stage0_file = f.name
                for token_len, name in chunk_results_sorted:
                    f.write(f"{name}:{token_len}\n")
            
            line_count = count_lines(stage0_file)
            stage0_queue.put(stage0_file)  # 放入跨进程队列
            # logger.info(f"进程 {multiprocessing.current_process().name} 完成块 {chunk_idx}，生成 {line_count} 条数据")
            # logger.info(f"�� 进程 {multiprocessing.current_process().name} 完成块 {chunk_idx}，有效样本 {processed_count}/{n_samples}")
            proc_status = "🟢" if processed_count==n_samples else "🟡"
            logger.info(f"{proc_status} 进程 {multiprocessing.current_process().name} 完成块 {chunk_idx}，有效样本 {processed_count}/{n_samples}")
            
            # 【关键】跨进程累加总数据量（使用Value原子操作）
            with global_total_counter.get_lock():
                global_total_counter.value += processed_count
                
            return stage0_file  # 返回生成的文件路径，用于后续清理
        
    except Exception as e:
        logger.error(f"❌ 进程 {multiprocessing.current_process().name} 处理失败: {str(e)}")
    finally:
        if processor:
            del processor
    return None


###
def main():
    global global_total_counter  # 引用全局变量
    processor = None   # 模型处理器实例
    stage0_files = []  # 记录所有stage0文件（用于验证和清理）
    stage1_files = []  # 记录所有stage1文件（用于最终合并）

    try:

        logger.info(f"💡 --------------开始数据处理流程--------------")
        
        # 1. 查找配对文件并写入临时文件（json和jpg文件名相同的样本）
        # base_names = find_paired_files(DEFAULT_DIRECTORY)    # DEFAULT_DIRECTORY 是原始数据存放位置（jpg 和 json）
        base_names = find_valid_json(DEFAULT_DIRECTORY)
        total_original = len(base_names)  # 原始样本总数
        logger.info(f"👉 找到 {total_original} 对原始样本文件")
        if total_original == 0:
            logger.warning("⚠️ 无原始样本，退出程序")
            return
        # 将配对文件名写入文件，用于后续分块读取
        write_base_names_to_file(base_names, OUTPUT_FILE)
        
        # 2. 初始化跨进程队列（用于传递stage0文件路径给合并线程）
        manager = Manager()  # 进程间共享队列需要用Manager
        stage0_queue = manager.Queue()
        stop_event = manager.Event()  # 跨进程停止信号

        # 跨进程计数器，用于统计总处理样本数（初始值0）
        global_total_counter = Value('i', 0)  # 'i'表示整数类型

        # 3 启动stage1合并线程（守护线程）
        stage1_thread = threading.Thread(
            target=stage1_merger,
            args=(stage0_queue, STAGE1_CHUNK, stage1_files, stop_event),
            daemon=True
        )
        stage1_thread.start()
        logger.info("💡 stage1合并线程已启动")

        # 4. 处理数据并生成stage0文件（每块数据单独处理并排序）
        # n_workers = 96 #get_adaptive_workers()

        # 4.1 读取所有数据块（准备分给多个进程)
        # chunk_size = chunk_size  # 每个进程处理的大chunk尺寸（根据内存调整）
        all_chunks = list(read_lines_in_chunks(OUTPUT_FILE, chunk_size))
        total_chunks = len(all_chunks)
        n_processes = min(multiprocessing.cpu_count(), total_chunks)
        logger.info(f"👉 划分为 {total_chunks} 个块，启动 {n_processes} 个进程处理")

        # 4.2 准备进程池参数（包含模型配置、队列等）
        process_args = [
            (
                idx + 1,  # chunk索引
                chunk,    # chunk数据
                CKPT_DIR, # 模型路径
                MIN_PIXELS,
                MAX_PIXELS,
                stage0_queue,  # 跨进程队列
            ) for idx, chunk in enumerate(all_chunks)
        ]
        
        # 4.3 启动进程池（进程数建议设为CPU核心数的1~2倍）
        with Pool(processes=n_processes) as process_pool:
            # 并行处理所有大chunk
            # stage0_files = process_pool.map(process_chunk, process_args)
            result = process_pool.map_async(process_chunk, process_args)
            try:
                stage0_files = result.get(timeout=TIME_OUT)  # 超时设置
            except multiprocessing.TimeoutError:
                logger.error("❌ 部分进程处理超时，强制终止")
                process_pool.terminate()
        
        # 过滤空结果
        stage0_files = [f for f in stage0_files if f is not None]
        logger.info(f"✅ 所有进程处理完成，共生成 {len(stage0_files)} 个stage0文件")  
        # 统计数据
        total_processed = global_total_counter.value  # 直接从全局变量获取  # 获取总处理样本数
        logger.info(f"👉 原始样本数: {total_original}, 有效处理样本数: {total_processed}")

        # 验证数据完整性
        if total_processed != total_original:
            logger.warning(f"❌ 数据不完整！原始 {total_original} 个，有效处理 {total_processed} 个，差异 {total_original - total_processed} 个")
        else:
            logger.info("✅ 数据完整性验证通过，所有样本均被有效处理")

        # 5. 等待处理完成（确保所有文件被合并）
        # 等待stage0队列所有文件被处理
        logger.info("🔄 等待stage0队列处理完成...")
        stage0_queue.join()  # 阻塞直到所有stage0文件被消费
        logger.info("💡 stage0队列所有文件已处理完毕")

        # 发送停止信号给stage1线程，强制处理剩余文件
        logger.info("💡 通知stage1线程停止并处理剩余文件...")
        stop_event.set()

        # 延长超时时间至60秒，确保大文件合并完成
        timeout_counter = 0
        while stage1_thread.is_alive() and timeout_counter < 60:
            logger.debug(f"🔄 等待stage1线程完成（{timeout_counter}/60秒）")
            threading.Event().wait(1)  # 等待1秒后重试
            timeout_counter += 1
        
        if stage1_thread.is_alive():
            logger.warning("⚠️ stage1线程超时未退出，可能存在异常（但已尝试强制合并剩余文件）")
        else:
            logger.info("💡 stage1线程已正常退出")

        # 验证stage1文件数量是否匹配（每10个stage0合并1个，不足10个也算1个）
        expected_stage1_count = (len(stage0_files) + STAGE1_CHUNK - 1) // STAGE1_CHUNK
        if len(stage1_files) != expected_stage1_count:
            logger.warning(f"⚠️ ℹ️  stage1文件数量异常！预期 {expected_stage1_count} 个，实际 {len(stage1_files)} 个")
        else:
            logger.info(f"✅ stage1文件数量验证通过: {len(stage1_files)} 个")

        # 6. 最终合并所有stage1文件到token_info_1.txt
        if not stage1_files:
            logger.warning("⚠️ 没有生成stage1文件，检查中间处理是否出错")
            return

        # 统计stage1文件总数据量
        stage1_total = sum(count_lines(f) for f in stage1_files)
        logger.info(f"ℹ️ 开始最终合并: {len(stage1_files)} 个stage1文件，总数据量: {stage1_total} 条")

        # 合并到最终文件
        final_path, final_lines = merge_files_by_token(stage1_files, TOKEN_INFO_FILE)

        if final_path and final_lines > 0:
            logger.info(f"✅ 最终结果文件生成完成: {TOKEN_INFO_FILE}，包含 {final_lines} 条数据")
            # 验证总数据量
            if final_lines != total_processed:
                logger.error(f"❌ 数据量不一致！处理总数据 {total_processed} 条，最终文件 {final_lines} 条")
            else:
                logger.info("✅💡 数据量验证通过，所有数据已正确写入最终文件")
        else:
            logger.error("❌ 最终文件合并失败")

        # 最终合并后再次验证
        if os.path.exists(TOKEN_INFO_FILE):
            final_count = count_lines(TOKEN_INFO_FILE)
            logger.info(f"ℹ️ 最终结果文件包含 {final_count} 条数据")
            if final_count != total_processed:
                logger.error(f"❌ 最终文件数据不完整！处理 {total_processed} 条，最终文件 {final_count} 条")
            else:
                logger.info("✅ 最终文件数据完整性验证通过")

    except Exception as e:
        logger.error(f"❌ 主流程错误: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        if processor:
            del processor

        # 确保停止信号被触发
        stop_event.set()

        if stage1_thread and stage1_thread.is_alive():
            stage1_thread.join(timeout=2)        
        
        # 等待最终文件写入完成
        threading.Event().wait(2)

        # 清理临时文件（保留最终文件）
        all_temp_files = stage0_files + stage1_files
        for fpath in all_temp_files:
            if fpath != str(TOKEN_INFO_FILE) and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug(f"已清理临时文件: {os.path.basename(fpath)}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败 {os.path.basename(fpath)}: {str(e)}")

        logger.info("程序执行完毕")


if __name__ == "__main__":
    main()

