# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import shutil
import sys

import numpy as np
import torch
from cuda import cudart

file_location = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(file_location, "Diffusion"))

import triton_python_backend_utils as pb_utils
from Diffusion.stable_diffusion_pipeline import StableDiffusionPipeline
from Diffusion.utilities import PIPELINE_TYPE


# Triton Python 后端入口类：实现 initialize / execute / finalize 三个生命周期钩子，
# 由 Triton 在模型加载、请求批处理和模型卸载时自动调用。
class TritonPythonModel:
    # 支持的 Stable Diffusion 版本 -> 管线类型 的映射表
    _KNOWN_VERSIONS = {"1.5": PIPELINE_TYPE.TXT2IMG, "xl-1.0": PIPELINE_TYPE.XL_BASE}

    # 设置各配置项的默认值，后续会被 config.pbtxt 中的参数覆盖
    def _set_defaults(self):
        self._batch_size = 1
        self._onnx_opset = 18
        self._image_height = 512
        self._image_width = 512
        self._seed = None
        self._version = "1.5"
        self._scheduler = None
        self._steps = 30
        self._force_engine_build = False

    # 读取单个字符串类型的配置参数，并转换为指定类型后赋给对应属性
    def _set_from_parameter(self, parameter, parameters, class_):
        value = parameters.get(parameter, None)
        if value is not None:
            value = value["string_value"]
            if value:
                setattr(self, "_" + parameter, class_(value))

    # 从 Triton 传入的 model_config（config.pbtxt 的 JSON 形式）中解析批大小与自定义参数
    def _set_from_config(self, model_config):
        model_config = json.loads(model_config)
        self._batch_size = int(model_config.get("max_batch_size", 1))
        if self._batch_size < 1:
            self._batch_size = 1

        config_parameters = model_config.get("parameters", {})

        if config_parameters:
            parameter_type_map = {
                "onnx_opset": int,
                "image_height": int,
                "image_width": int,
                "steps": int,
                "seed": int,
                "scheduler": str,
                "guidance_scale": float,
                "version": str,
                "force_engine_build": bool,
            }

            for parameter, parameter_type in parameter_type_map.items():
                self._set_from_parameter(parameter, config_parameters, parameter_type)

    # Triton 生命周期钩子：模型加载时调用一次，负责解析配置、创建扩散管线并加载 TensorRT 引擎
    def initialize(self, args):
        self._set_defaults()
        self._set_from_config(args["model_config"])

        if self._version not in TritonPythonModel._KNOWN_VERSIONS:
            raise Exception(
                f"Invalid Stable Diffusion Version: {self._version}, choices: {list(TritonPythonModel._KNOWN_VERSIONS.keys())}"
            )

        self._model_instance_device_id = int(args["model_instance_device_id"])

        # 创建扩散模型管线：use_cuda_graph=True 开启 CUDA Graph 以降低每次推理的启动开销
        self._pipeline = StableDiffusionPipeline(
            pipeline_type=TritonPythonModel._KNOWN_VERSIONS[self._version],
            max_batch_size=self._batch_size,
            use_cuda_graph=True,
            version=self._version,
            denoising_steps=self._steps,
        )

        # 按"版本-批大小"约定目录名，引擎/权重/ONNX 文件都存在模型版本目录下
        model_directory = os.path.join(args["model_repository"], args["model_version"])
        engine_dir = os.path.join(
            model_directory, f"{self._version}-engine-batch-size-{self._batch_size}"
        )
        framework_model_dir = os.path.join(
            model_directory, f"{self._version}-pytorch_model"
        )
        onnx_dir = os.path.join(model_directory, f"{self._version}-onnx")

        # force_engine_build 置为真时，删除已有产物以触发完整重建
        if self._force_engine_build:
            shutil.rmtree(engine_dir, ignore_errors=True)
            shutil.rmtree(framework_model_dir, ignore_errors=True)
            shutil.rmtree(onnx_dir, ignore_errors=True)

        # 当前后端实现只支持 GPU 0 上的单个实例
        if self._model_instance_device_id != 0:
            raise Exception("Only device id 0 is currently supported")

        # 加载或构建 TensorRT 引擎；static_batch=True 表示引擎按固定批大小编译
        self._pipeline.loadEngines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            onnx_opset=self._onnx_opset,
            opt_batch_size=self._batch_size,
            opt_image_height=self._image_height,
            opt_image_width=self._image_width,
            static_batch=True,
        )
        # 一次性分配引擎所需的最大共享设备内存，供所有引擎复用
        _, shared_device_memory = cudart.cudaMalloc(
            self._pipeline.calculateMaxDeviceMemory()
        )
        self._pipeline.activateEngines(shared_device_memory)
        # 加载文本/图像等资源；seed 非空时生成结果可复现
        self._pipeline.loadResources(
            self._image_height, self._image_width, self._batch_size, seed=self._seed
        )
        self._logger = pb_utils.Logger

    # Triton 生命周期钩子：模型卸载时调用，释放引擎与显存资源
    def finalize(self):
        self._pipeline.teardown()

    # Triton 生命周期钩子：推理主入口。requests 是 Triton 动态批处理后下发的一批请求
    def execute(self, requests):
        responses = []
        prompts = []
        negative_prompts = []
        prompts_per_request = []
        image_results = []
        for request in requests:
            # 从请求中读取 prompt 输入张量，逐条解码为文本
            prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()

            for prompt in prompt_tensor:
                prompts.append(prompt[0].decode())

            # negative_prompt 为可选输入；未提供时用空字符串代替
            negative_prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "negative_prompt"
            )

            if not negative_prompt_tensor:
                negative_prompts.extend([""] * len(prompt_tensor))
            else:
                negative_prompt_tensor = negative_prompt_tensor.as_numpy()
                for negative_prompt in negative_prompt_tensor:
                    negative_prompts.append(negative_prompt[0].decode())
            prompts_per_request.append(len(prompt_tensor))
        num_requests = len(requests)
        num_prompts = len(prompts)
        # 引擎按固定批大小编译，提示词数量不足一个批次时用空字符串补齐
        remainder = self._batch_size - (num_prompts % self._batch_size)
        self._logger.log_info(f"Client Requests in Batch:{num_requests}")
        self._logger.log_info(f"Prompts in Batch:{num_prompts}")
        if remainder < self._batch_size:
            prompts.extend([""] * remainder)
            negative_prompts.extend([""] * remainder)
        num_prompts = len(prompts)
        # 按固定批大小切分，逐批送入引擎执行推理
        for batch in range(0, num_prompts, self._batch_size):
            (images, walltime_ms) = self._pipeline.infer(
                prompts[batch : batch + self._batch_size],
                negative_prompts[batch : batch + self._batch_size],
                self._image_height,
                self._image_width,
                save_image=False,
            )
            # 图像后处理：把管线输出的 [-1, 1] 范围张量映射回 [0, 255] 的 uint8 像素值，
            # 并从 (N,C,H,W) 调整为 (N,H,W,C) 便于保存和展示
            images = (
                ((images + 1) * 255 / 2)
                .clamp(0, 255)
                .detach()
                .permute(0, 2, 3, 1)
                .round()
                .type(torch.uint8)
                .cpu()
                .numpy()
            )
            image_results.extend(images)

        # 按每个请求原始携带的提示词数量，把生成结果切分回对应的响应，
        # 保证一个请求对应一个 InferenceResponse
        result_index = 0
        for num_prompts_in_request in prompts_per_request:
            generated_images = []
            for image_result in image_results[
                result_index : result_index + num_prompts_in_request
            ]:
                generated_images.append(image_result)
            # 构造 Triton 响应对象，输出名为 "generated_image" 的张量
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(generated_images, dtype=np.uint8),
                    )
                ]
            )
            responses.append(inference_response)
            result_index += num_prompts_in_request

        return responses
