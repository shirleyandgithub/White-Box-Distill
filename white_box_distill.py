# encoding=utf-8

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor  # 确保transformer为最新版本
from qwen_vl_utils import process_vision_info  # 需要安装qwen-vl-utils包
import bitsandbytes as bnb  # 需要安装bitsandbytes

TEACHER_PATH = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
STUDENT_PATH = "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct"

# 手动指定设备
DEVICE_TEACHER = "cuda:0"  # Teacher独占一张卡
DEVICE_STUDENT = "cuda:1"  # Student独占一张卡

# 加载Teacher模型
print("\n加载Teacher模型 (冻结参数, BF16)...")

# Teacher不需要梯度，使用eval模式节省显存
teacher = AutoModelForImageTextToText.from_pretrained(
    TEACHER_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE_TEACHER,
    trust_remote_code=True
).eval()

# 冻结Teacher的所有参数
for param in teacher.parameters():
    param.requires_grad = False

# 加载Student模型
print("\n加载Student模型 (训练模式, BF16)...")

student = AutoModelForImageTextToText.from_pretrained(
    STUDENT_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE_STUDENT,
    trust_remote_code=True,
    use_cache=False
).train()

# 开启梯度检查点
student.gradient_checkpointing_enable()

print("\n加载处理器 (极速演示模式: 256x256)...")
processor = AutoProcessor.from_pretrained(
    TEACHER_PATH,
    min_pixels=256 * 256,
    max_pixels=256 * 256,
    trust_remote_code=True,
    use_fast=True
)

print("\n准备多模态输入数据...")
# 模拟一条多模态数据
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "详细描述这张图片的内容。"},
        ],
    }
]

# 预处理数据
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# 蒸馏循环
print("\n跨卡蒸馏训练 (Teacher -> Student)...")

# 定义优化器 (只优化Student)
optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=1e-5)
TEMPERATURE = 2.0  # 蒸馏温度：软化概率分布

# 模拟10个训练步数
for step in range(1, 11):
    # Teacher推理 (在GPU 0): 把数据搬到GPU 0
    inputs_t = {k: v.to(DEVICE_TEACHER) for k, v in inputs.items()}

    with torch.no_grad():
        teacher_outputs = teacher(**inputs_t)
        # 获取Teacher的Logits并跨卡搬运到GPU 1，其中detach()是为了断开计算图，to(DEVICE_STUDENT)是物理搬运
        teacher_logits = teacher_outputs.logits.detach().to(DEVICE_STUDENT)

    # Student推理(在 GPU 1): 把数据搬到GPU 1
    inputs_s = {k: v.to(DEVICE_STUDENT) for k, v in inputs.items()}
    student_outputs = student(**inputs_s)
    student_logits = student_outputs.logits

    # 计算Loss(在GPU 1): 对齐词表大小，防止Teacher和Student词表最后一维有细微差异
    vocab_size = min(teacher_logits.size(-1), student_logits.size(-1))

    # 截取对齐
    s_logits_aligned = student_logits[..., :vocab_size]
    t_logits_aligned = teacher_logits[..., :vocab_size]

    # KL散度计算
    loss_distill = F.kl_div(
        F.log_softmax(student_logits[..., :vocab_size] / TEMPERATURE, dim=-1),
        F.softmax(teacher_logits[..., :vocab_size] / TEMPERATURE, dim=-1),
        reduction='batchmean'
    ) * (TEMPERATURE ** 2)

    # 反向传播与更新
    optimizer.zero_grad()
    loss_distill.backward()
    optimizer.step()

    print(f"Step {step}: Distillation Loss = {loss_distill.item():.6f}")

    # 显存清理 (防止碎片堆积)
    del teacher_logits, student_logits, student_outputs, loss_distill
    torch.cuda.empty_cache()

# 保存蒸馏后的模型
SAVE_PATH = "/root/autodl-tmp/student_distilled"
print(f"保存蒸馏后的模型到 {SAVE_PATH} ...")
student.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH) # 把处理器配置也存进去，方便加载

print("\n蒸馏结束！")



