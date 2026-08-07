# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

import torch
from model import STRModel

# 创建 PyTorch 模型对象（单通道输入、512 维特征、37 类字符）
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# 从外部文件加载模型权重
state = torch.load("downloads/None-ResNet-None-CTC.pth")
# 去掉权重名中的 "module." 前缀（DataParallel 训练产物的兼容处理）
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# 通过 trace 方式把 PyTorch 模型导出为 ONNX 文件
model_directory = Path("model_repository/text_recognition/1/")
model_directory.mkdir(parents=True, exist_ok=True)
trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(
    model,
    trace_input,
    model_directory / "model.onnx",
    verbose=True,
    # 把输入和输出的第 0 维声明为动态维度，这样 Triton 才能对多个请求合批
    dynamic_axes={"input.1": [0], "308": [0]},
)
