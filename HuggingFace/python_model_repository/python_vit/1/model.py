# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import ViTImageProcessor, ViTModel


class TritonPythonModel:
    # Triton 加载模型时调用：加载 ViT 图像处理器和 ViT 模型本体（仅执行一次）
    def initialize(self, args):
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # 每个推理请求都会调用：执行预处理 + 模型前向，返回编码后的特征张量
    def execute(self, requests):
        responses = []
        for request in requests:
            # 按名称从请求中取出输入张量
            inp = pb_utils.get_input_tensor_by_name(request, "image")
            # 去掉 batch 维度并把 HWC 布局转为 CHW
            input_image = np.squeeze(inp.as_numpy()).transpose((2, 0, 1))
            # 图像处理器负责归一化、切分 patch 等 ViT 标准预处理
            inputs = self.feature_extractor(images=input_image, return_tensors="pt")

            # 模型前向传播，得到编码器输出
            outputs = self.model(**inputs)

            # 把 last_hidden_state 转为 numpy 并封装成 Triton 响应
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "last_hidden_state", outputs.last_hidden_state.detach().numpy()
                    )
                ]
            )
            responses.append(inference_response)
        return responses
