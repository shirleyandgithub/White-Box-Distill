**资源分配**:<br>
GPU 0加载Qwen2.5-VL-7B，开启eval( )模式(不计算梯度)，BF精度约占16GB显存，一张4090(24G)可以满足;<br>
GPU 1加载Qwen2.5-VL-3B，开启train( )模式，进行反向传播;<br><br><br>


**数据流向(跨卡传输)**:<br>
Data → GPU 0(Teacher Inference) → Output Logits → 跨卡搬运(.to('cuda:1')) → GPU 1(计算Loss) → Student Update;<br>
