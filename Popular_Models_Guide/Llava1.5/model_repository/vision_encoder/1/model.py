# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tensorrt as trt
import torch
import triton_python_backend_utils as pb_utils


# Triton Python 后端入口类：封装 Llava1.5 的视觉编码器 TensorRT 引擎，
# 接收原始图像张量，输出图片特征（供 llava-1.5 模型作视觉 token 使用）。
class TritonPythonModel:
    # 生命周期钩子：模型加载时调用，从 TRT_ENGINE_LOCATION 环境变量加载视觉编码器引擎
    def initialize(self, args):
        # 依据模型的实例类型（GPU/CPU）确定运行设备
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        # 反序列化 TensorRT 引擎：读取二进制引擎文件并创建执行上下文
        self.logger = trt.Logger(trt.Logger.ERROR)
        engine_path = os.getenv("TRT_ENGINE_LOCATION")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # 枚举引擎的全部输入/输出张量，记录名称、数据类型与形状，供后续绑定内存
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True

            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            # 动态形状输入（维度含 -1）需要从优化 profile 中取形状
            if shape[0] < 0:
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                # 取 profile 的 *min* 作为绑定形状，可选 [min,opt,max]
                self.context.set_input_shape(name, profile_shape[0])
                shape = self.context.get_tensor_shape(name)

            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": None,
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    # 推理主入口：把请求中的图像张量拷入 GPU，执行引擎，返回 "features" 输出张量
    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            allocations = []
            # 按引擎输出形状预分配输出张量（零初始化）
            output = torch.asarray(
                np.zeros(self.outputs[0]["shape"], self.outputs[0]["dtype"]),
                device=self.device,
            )
            # 把请求中的图像数据搬移到 GPU 设备内存
            input_tensor = torch.asarray(
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy(),
                device=self.device,
            )
            # 绑定输入/输出的设备内存指针，交给引擎执行（execute_v2 零拷贝方式）
            self.inputs[0]["allocation"] = input_tensor.data_ptr()
            allocations.append(input_tensor.data_ptr())
            self.outputs[0]["allocation"] = output.data_ptr()
            allocations.append(output.data_ptr())
            self.context.execute_v2(allocations)
            # 输出张量拷回 CPU，并通过 DLPack 封装为 Triton 输出张量
            out_tensor = pb_utils.Tensor.from_dlpack("features", output.cpu())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
            # 清理绑定，避免悬空指针
            self.inputs[0]["allocation"] = None
            self.outputs[0]["allocation"] = None
        return responses
