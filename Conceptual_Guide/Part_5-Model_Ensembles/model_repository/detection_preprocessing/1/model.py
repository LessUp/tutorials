# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import io
import json

import numpy as np
import torch
import torchvision.transforms as transforms

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from PIL import Image


# Python backend 模型入口：类名必须是 TritonPythonModel，
# 把图片解码 + 缩放归一化的前处理从客户端搬到 Triton 服务端执行
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    # 模型加载时调用一次（可选）：解析 model_config，准备输出数据类型
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # 必须解析 model_config（这里拿到的是 JSON 字符串，需要手动解析）
        model_config = json.loads(args["model_config"])

        # 从模型配置中取出输出张量的配置
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "detection_preprocessing_output"
        )

        # 把 Triton 数据类型转换成对应的 numpy 类型，供输出张量使用
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # 每个 Python backend 都必须遍历所有请求，
        # 并为每一条请求构造一个 pb_utils.InferenceResponse
        for request in requests:
            # 从请求中按名称取出输入张量（集成模型传入的原始图片字节）
            in_0 = pb_utils.get_input_tensor_by_name(
                request, "detection_preprocessing_input"
            )

            # 图片加载器：裁剪到 32 的倍数尺寸并转为张量（对齐模型输入约束）
            def image_loader(image):
                [h, w] = image.size
                resize_w = (w // 32) * 32
                resize_h = (h // 32) * 32

                center_crop = resize_h if resize_w > resize_h else resize_w
                loader = transforms.Compose(
                    [transforms.CenterCrop(center_crop), transforms.ToTensor()]
                )

                im = loader(image)
                im = torch.unsqueeze(im, 0)
                return im.permute(0, 2, 3, 1)

            img = in_0.as_numpy()

            # 把输入字节解码成 PIL 图片，再走前处理流水线并还原 0-255 数值范围
            image = Image.open(io.BytesIO(img.tobytes()))
            img_out = image_loader(image)
            img_out = np.array(img_out) * 255.0

            # 构造输出张量，名称与 config.pbtxt 中定义的输出一致
            out_tensor_0 = pb_utils.Tensor(
                "detection_preprocessing_output", img_out.astype(output0_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        # 返回的响应列表长度必须与请求列表一致
        return responses

    # 模型卸载时调用一次（可选）：做一些清理工作
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
