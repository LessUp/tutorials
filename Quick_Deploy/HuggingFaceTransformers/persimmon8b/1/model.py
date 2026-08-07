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
import os

# 在导入 transformers 之前设置模型缓存目录，避免每次启动都重新下载模型
os.environ[
    "TRANSFORMERS_CACHE"
] = "/opt/tritonserver/model_repository/persimmon8b/hf_cache"

import json

import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    # 初始化：读取模型配置参数，加载分词器与文本生成 pipeline
    def initialize(self, args):
        self.logger = pb_utils.Logger
        # 解析 model config 中的自定义参数（huggingface_model、max_output_length）
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        default_hf_model = "adept/persimmon-8b-base"
        default_max_gen_length = "15"
        # 从配置中读取用户指定的 HuggingFace 模型名，未指定时使用默认模型
        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )
        # 从配置中读取最大生成长度，未指定时默认 15 个 token
        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get(
                "string_value", default_max_gen_length
            )
        )

        self.logger.log_info(f"Max output length: {self.max_output_length}")
        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")
        # 加载与模型配套的分词器（假定同名仓库中提供）
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
        # 构建文本生成 pipeline：FP16 精度节省显存，device_map="auto" 自动分配设备
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=hf_model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    # 每个推理请求都会调用：取出输入文本，生成并返回文本响应
    def execute(self, requests):
        responses = []
        for request in requests:
            # 按名称从请求中取出输入张量（text_input）
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")

            self.logger.log_info(f"Generating sequences for text_input: {prompt}")
            response = self.generate(prompt)
            responses.append(response)

        return responses

    # 核心生成逻辑：调用 pipeline 生成单个序列，封装为 Triton 响应
    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            max_length=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_tensors = []
        texts = []
        for i, seq in enumerate(sequences):
            text = seq["generated_text"]
            self.logger.log_info(f"Sequence {i+1}: {text}")
            texts.append(text)

        # 以 object 数组承载文本字符串，封装为 Triton 输出张量
        tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
        output_tensors.append(tensor)
        response = pb_utils.InferenceResponse(output_tensors=output_tensors)
        return response

    # Triton 卸载模型时调用：清理资源
    def finalize(self):
        print("Cleaning up...")
