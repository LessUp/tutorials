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
import requests as rq
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


# Triton Python 后端入口类：编排 Llava1.5 的多模态推理流程。
# 它先调用同仓库下的 vision_encoder 子模型提取图片特征，
# 再把特征作为视觉 token 嵌入提示词，交给 tensorrt_llm 模型做文本生成（即 BLS 编排模式）。
class TritonPythonModel:
    # 生命周期钩子：模型加载时调用，从 HF_LOCATION 环境变量指向的目录加载图像处理器与分词器
    def initialize(self, args):
        # 环境变量未设置时回退到模型目录
        HF_LOCATION = os.getenv("HF_LOCATION", pb_utils.get_model_dir())
        self.image_processor = AutoProcessor.from_pretrained(HF_LOCATION)
        self.logger = pb_utils.Logger
        self.tokenizer = AutoTokenizer.from_pretrained(HF_LOCATION)
        # Llava1.5 的词表大小为 32064；视觉 token 从词表末尾之后开始编号，避免与文本 token 冲突
        self.vocab_size = 32064
        self.max_input_len = 2048

    # 把提示词按 <image> 占位符切分，中间插入 num_visual_tokens 个视觉 token 的 ID 区间
    def _tokenize(self, prompt, num_visual_tokens):
        chunks = prompt.split("<image>")
        assert len(chunks) == 2, "Only support exactly one image per prompt"

        return (
            self.tokenizer.encode(chunks[0])
            + list(range(self.vocab_size, self.vocab_size + num_visual_tokens))
            + self.tokenizer.encode(chunks[1])[self.tokenizer.add_bos_token :]
        )

    # 从请求中按名称读取输入张量；请求未携带该输入时返回默认值
    def _parse_input(self, request, input_name, default=None):
        input = pb_utils.get_input_tensor_by_name(request, input_name)
        if input is not None:
            return input.as_numpy()[0]

        return default

    # 提取图片特征：下载图片并调用 vision_encoder 子模型（详见下方 docstring）
    def _extract_image_features(self, image, prompt):
        """
        Extracts features from an image using a vision encoder model. This
        helper function loads an image from the provided URL using the
        `requests` library. The image is then converted to RGB format using the
        `PIL` library. Next, the image is processed using `transformers`'
        image AutoProcessor (defined in `initialize` method), which takes the
        prompt and the image as inputs.

        An inference request object is created for the `vision_encoder` model,
        which returns image features to pass to LLM.


        Parameters
        ----------
        - image (str): The URL or path to the image to be processed.
        - prompt (str): A textual prompt that may be used by the image processor.

        Returns
        -------
        - torch.Tensor: A tensor containing the extracted image features.
        """

        # 从 URL 流式下载图片并转为 RGB
        pil_image = Image.open(rq.get(image, stream=True).raw).convert("RGB")
        # 用 AutoProcessor 把图片预处理成模型期望的像素张量（float16 以省显存）
        image = self.image_processor(
            text=prompt, images=pil_image, return_tensors="np"
        )["pixel_values"].astype(np.float16)
        # 构造对 vision_encoder 子模型的 BLS 推理请求（同服务器内调用，不走网络）
        infer_request = pb_utils.InferenceRequest(
            model_name="vision_encoder",
            requested_output_names=["features"],
            inputs=[pb_utils.Tensor("image", image)],
        )
        vision_response = infer_request.exec()
        image_features = pb_utils.get_output_tensor_by_name(vision_response, "features")
        return torch.from_dlpack(image_features.as_numpy())

    # 组装传给 tensorrt_llm 模型的输入：文本 token、采样参数与图片特征（详见下方 docstring）
    def _prepare_llm_inputs(self, request, image_features, prompt):
        """
        Prepares inputs for the language model based on the parameters in the
        request, image features, and prompt. It tokenizes prompt,
        extracts and processes additional parameters from the request:
            - max_tokens: Maximum number of tokens to generate (default: 50)
            - temperature: Controls randomness in generation (default: 0.5)
            - top_k: Top K sampling parameter (default: 1)
            - frequency_penalty: Penalizes frequent tokens (default: 0.7)
            - seed: Random seed for generation (default: 10)

        Final llm input dictionary is combined out of all processed parameters,
        prompt's tokens and image features. The latter will be passed to llm
        through `prompt_embedding_table`.

        Parameters
        ----------
        - request: The original request object containing additional parameters.
        - image_features (list): A list containing image feature tensors.
        - prompt (str): The text prompt to be processed.

        Returns
        -------
        - dict: A dictionary containing all the prepared inputs for the language model.
        """
        # 生成含视觉 token 的完整 token 序列，并做输入长度上限校验
        input_ids = self._tokenize(prompt, len(image_features[0]))
        input_ids = np.array(input_ids, dtype=np.int32)
        input_len = input_ids.shape[0]
        if input_len > self.max_input_len:
            return pb_utils.TritonError(
                f"Input length ({input_len:d}) exceeds limit ({self.max_input_len:d})"
            )
        # 从请求中读取采样参数（缺失时用默认值）
        max_tokens = self._parse_input(request, "max_tokens", default=50)
        temperature = self._parse_input(request, "temperature", default=0.5)
        top_k = self._parse_input(request, "top_k", default=1)
        frequency_penalty = self._parse_input(request, "frequency_penalty", default=0.7)
        seed = self._parse_input(request, "seed", default=10)
        # 视觉特征通过 prompt_embedding_table 传给 TRT-LLM：
        # 视觉 token 的嵌入由该表直接提供，而非词表查找——这是多模态接入 LLM 的通用机制
        embedding_args = {
            "prompt_vocab_size": np.array(
                [[image_features[0].shape[0]]], dtype=np.uint32
            ),
            "prompt_embedding_table": np.expand_dims(image_features[0], 0).astype(
                np.float16
            ),
        }

        # 组装 TRT-LLM 期望的输入字典：形状统一为 (1, 1) 或 (1, 特征数)
        return {
            "input_ids": np.expand_dims(input_ids, 0),
            "input_lengths": np.array([[input_len]], dtype=np.int32),
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "temperature": np.array([[temperature]], dtype=np.float32),
            "runtime_top_k": np.array([[top_k]], dtype=np.int32),
            "frequency_penalty": np.array([[frequency_penalty]], dtype=np.float32),
            "end_id": np.array([[self.tokenizer.eos_token_id]], dtype=np.int32),
            "random_seed": np.array([[seed]], dtype=np.uint64),
            "streaming": np.array([[1]], dtype=np.bool_),
            **embedding_args,
        }

    # 把输入发给 tensorrt_llm 模型做流式生成，并聚合成最终响应（详见下方 docstring）
    def _prepare_llm_response(self, llm_request_inputs):
        """
        Prepares the response from the language model based on the provided
        inputs. Creates a `pb_utils.InferenceRequest` object with passed
        `llm_request_inputs` to send to a decoupled TensorRTLLM model.
        For each response from the language model:
            - Checks for errors and raise an exception if any are found.
            - Extracts the "output_ids" tensor from the response.
            - Determines the finish reason based on the presence of the
              end-of-sequence token or reaching the maximum length.
            - Appends the generated token IDs to `output_ids`.
            - If the finish reason is determined, decodes the output IDs to text
              and prepares the final response.

        The final response includes the generated text, finish reason,
        completion tokens, prompt tokens, and total tokens.

        Parameters
        ----------
        - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

        Returns
        -------
        - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
        """
        # 向 tensorrt_llm 模型发起 BLS 请求；decoupled=True 开启解耦（流式）模式，
        # exec 会逐次返回每个流式片段
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=[pb_utils.Tensor(k, v) for k, v in llm_request_inputs.items()],
        )
        output_ids, output_len = [], 0
        max_len = llm_request_inputs["request_output_len"][0][0]

        for llm_response in llm_request.exec(decoupled=True):
            if llm_response.has_error():
                raise pb_utils.TritonModelException(llm_response.error().message())
            # 取出本片段的输出 token ID
            stream_output_ids = (
                pb_utils.get_output_tensor_by_name(llm_response, "output_ids")
                .as_numpy()
                .flatten()
                .tolist()
            )
            # 判定结束原因：出现 EOS 记号或生成到最大长度
            finish_reason = ""
            if len(stream_output_ids) == 0 or (
                len(stream_output_ids) != 0
                and stream_output_ids[-1] == self.tokenizer.eos_token_id
            ):
                finish_reason = "stop"
            output_ids += stream_output_ids
            if len(output_ids) >= max_len:
                finish_reason = "length"
                output_ids = output_ids[:max_len]
            last_response = finish_reason != ""
            output_len = len(output_ids)
            if last_response:
                # 结束：把 token 解码成文本，构造带统计信息的最终响应
                output_text = self.tokenizer.decode(output_ids).strip()
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("text", np.array([output_text], np.object_)),
                        pb_utils.Tensor(
                            "finish_reason", np.array([finish_reason], np.object_)
                        ),
                        pb_utils.Tensor(
                            "completion_tokens", np.array([output_len], np.int32)
                        ),
                        pb_utils.Tensor(
                            "prompt_tokens",
                            np.array([llm_request_inputs["input_lengths"]], np.int32),
                        ),
                        pb_utils.Tensor(
                            "total_tokens",
                            np.array(
                                [output_len + llm_request_inputs["input_lengths"]],
                                np.int32,
                            ),
                        ),
                    ]
                )
                return response
        return None

    # 推理主入口：逐个处理请求，串起"提特征 → 组输入 → LLM 生成"三步。
    # 使用解耦（decoupled）响应发送方式，通过 response_sender 异步回传结果
    def execute(self, requests):
        for request in requests:
            # 获取与请求绑定的响应发送器（解耦模式要求显式发送并标记最终响应）
            response_sender = request.get_response_sender()
            # 读取图片输入：可以是 URL 字符串，也可能是 gRPC 传输后的 bytes，统一解码
            image = (
                pb_utils.get_input_tensor_by_name(request, "image")
                .as_numpy()
                .flatten()
                .tolist()
            )
            if isinstance(image[0], bytes):
                image = image[0].decode("utf-8")

            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            # Step 1. 加载图片并调用 vision_encoder 子模型提取图像特征
            image_features = self._extract_image_features(image, prompt)
            # Step 2. 把图像特征、提示词与请求中的采样参数组合成 LLM 输入
            llm_request_inputs = self._prepare_llm_inputs(
                request, image_features, prompt
            )
            # 输入超长时直接以错误响应结束，不再调用 LLM
            if isinstance(llm_request_inputs, pb_utils.TritonError):
                error = pb_utils.InferenceResponse(error=llm_request_inputs)
                response_sender.send(
                    error, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                return
            # Step 3. 把组装好的输入交给 tensorrt_llm 模型生成文本
            llm_response = self._prepare_llm_response(llm_request_inputs)
            if llm_response is not None:
                response_sender.send(
                    llm_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        return None
