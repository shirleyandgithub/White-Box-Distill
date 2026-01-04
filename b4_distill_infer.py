# encoding=utf-8

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor  # 确保transformer为最新版本
from qwen_vl_utils import process_vision_info  # 需要安装qwen-vl-utils包

MODEL_PATH = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
IMAGE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
PROMPT = "详细描述这张图片。"

# 加载模型与处理器
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256*256,
    max_pixels=1024*1024,
    trust_remote_code=True,
    use_fast=True
)

# 数据预处理
messages = [
    {
        "role": "user",
        "content": [{"type": "image",
                     "image": IMAGE},
                    {"type": "text",
                     "text": PROMPT}]
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt").to("cuda")

# 推理生成并打印
generated_ids = model.generate(**inputs, max_new_tokens=256)

# 手动切掉输入的长度
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 仅解码希望显示的的部分
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(output_text)



