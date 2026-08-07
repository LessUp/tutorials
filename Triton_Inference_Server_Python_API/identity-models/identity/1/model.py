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

# ============================================================================
# Triton Python backend 的 identity（身份）模型：
# 把输入张量原样复制到输出，用于验证 Triton 对各种数据类型的处理能力。
# ============================================================================

import json

import triton_python_backend_utils as pb_utils

try:
    import cupy
except ImportError:
    cupy = None


# Triton Python backend 模型入口类：Triton 按 config.pbtxt 中的配置实例化
# 该类，并在推理请求到达时调用 execute()。
class TritonPythonModel:
    # 自动补全模型配置：为 Triton 支持的所有数据类型生成对应的输入/输出定义
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = []
        outputs = []
        dims = [-1, -1]
        optional = True
        config = auto_complete_model_config.as_dict()

        # 遍历 Triton 支持的每种数据类型，生成 <type>_input / <type>_output 张量
        for data_type in pb_utils.TRITON_STRING_TO_NUMPY.keys():
            type_name = data_type.split("_")[1].lower()
            input_name = f"{type_name}_input"
            output_name = f"{type_name}_output"
            inputs.append(
                {
                    "name": input_name,
                    "data_type": data_type,
                    "dims": dims,
                    "optional": optional,
                }
            )
            outputs.append({"name": output_name, "data_type": data_type, "dims": dims})

        for input_ in inputs:
            auto_complete_model_config.add_input(input_)
        for output in outputs:
            auto_complete_model_config.add_output(output)

        # 不启用动态批处理（max_batch_size=0），默认按非解耦（decoupled）模式处理
        auto_complete_model_config.set_max_batch_size(0)

        auto_complete_model_config.set_model_transaction_policy({"decoupled": False})

        # 若模型参数中显式开启 decoupled，则改用解耦事务策略
        if "parameters" in config and "decoupled" in config["parameters"]:
            if config["parameters"]["decoupled"]["string_value"] == "True":
                auto_complete_model_config.set_model_transaction_policy(
                    {"decoupled": True}
                )

        return auto_complete_model_config

    # 模型加载时调用一次：解析模型配置，判断是否为解耦模式、是否请求 GPU 内存
    def initialize(self, args):
        self._model_config = json.loads(args["model_config"])
        self._decoupled = self._model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )
        self._request_gpu_memory = False
        # 可通过模型参数 request_gpu_memory=True 让输出张量直接分配在 GPU 上
        if "parameters" in self._model_config:
            parameters = self._model_config["parameters"]
            if (
                "request_gpu_memory" in parameters
                and parameters["request_gpu_memory"]["string_value"] == "True"
            ):
                self._request_gpu_memory = True

    # 解耦模式执行：每个请求通过 response_sender 独立发送响应；
    # 支持多次 send（流式输出），结束后用 TRITONSERVER_RESPONSE_COMPLETE_FINAL 标记完成
    def execute_decoupled(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            output_tensors = []
            # 遍历请求的每个输入张量，复制一份并改名为 <type>_output
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()
                output_tensor = pb_utils.Tensor(
                    input_tensor.name().replace("input", "output"), input_value
                )
                output_tensors.append(output_tensor)
            sender.send(pb_utils.InferenceResponse(output_tensors=output_tensors))
            # 标记响应已发送完毕（流式场景中可在此之前多次 send 中间结果）
            sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None

    # 常规（非解耦）执行入口：一次处理一批请求，返回响应列表
    def execute(self, requests):
        if self._decoupled:
            return self.execute_decoupled(requests)
        responses = []
        for request in requests:
            output_tensors = []
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()

                # 可选：把输入拷贝到 GPU（cupy）后再返回，验证 GPU 张量路径
                if self._request_gpu_memory:
                    input_value = cupy.array(input_value)

                    output_tensor = pb_utils.Tensor.from_dlpack(
                        input_tensor.name().replace("input", "output"), input_value
                    )
                else:
                    # 普通路径：把 numpy 张量直接包装为输出张量
                    output_tensor = pb_utils.Tensor(
                        input_tensor.name().replace("input", "output"), input_value
                    )
                output_tensors.append(output_tensor)

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses
