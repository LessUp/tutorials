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
import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# 单条请求的生成状态：提示词长度、已生成的 token 序列、生成上限等
class State:
    def __init__(self):
        self.prompt_tokens_len = 0
        self.tokens = []
        self.max_tokens = 0
        self.ignore_eos = False


# 迭代调度版 GPT-2：每次被调度只生成 1 个 token，
# 通过 RELEASE_RESCHEDULE 让 Triton 把未完成的请求重新调度回来，
# 新到达的请求可以插入同一批，实现 inflight batching
class TritonPythonModel:
    # 初始化：按模型实例的设备加载 GPT-2 模型
    def initialize(self, args):
        # 以 correlation_id 为键保存各请求的生成状态
        self.state = {}
        device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"

        # 加载 GPT-2 模型与分词器
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # 自动补全模型配置：声明输入输出，并启用 decoupled（解耦）模式
    @staticmethod
    def auto_complete_config(config):
        inputs = [
            {
                "name": "text_input",
                "data_type": "TYPE_STRING",
                "dims": [1],
            }
        ]
        outputs = [{"name": "text_output", "data_type": "TYPE_STRING", "dims": [1]}]

        for input in inputs:
            config.add_input(input)
        for output in outputs:
            config.add_output(output)

        # 启用解耦模式：响应不要求与请求一一同步返回，流式逐 token 发送
        transaction_policy = {"decoupled": True}
        config.set_model_transaction_policy(transaction_policy)
        config.set_max_batch_size(8)

        return config

    # 把一批请求的输入收集起来，生成合批后的 input_ids 与 attention_mask
    def create_batch(self, requests):
        """
        Gathers input tensors from the requests and returns processed input tensors.

        Args:
            requests (list): A list of requests containing input tensors.

        Returns:
            input_ids (torch.Tensor): A tensor containing the processed input IDs.
            attention_mask (torch.Tensor): A tensor containing the attention mask.
        """

        input_ids = []
        for request in requests:
            # 读取请求携带的输入文本
            input_tensor = str(
                pb_utils.get_input_tensor_by_name(request, "text_input")
                .as_numpy()
                .item(),
                encoding="utf-8",
            )
            # correlation_id 用于区分同一条请求的不同调度轮次（迭代调度核心机制）
            correlation_id = (
                pb_utils.get_input_tensor_by_name(request, "correlation_id")
                .as_numpy()
                .item()
            )
            # start 标志表示这是该请求的第一次调度，需要初始化状态
            start = (
                pb_utils.get_input_tensor_by_name(request, "start").as_numpy().item()
            )
            if start:
                state = State()
                state.tokens = self.tokenizer(
                    input_tensor, return_tensors="pt", padding=True
                )["input_ids"][0].to(self.device)
                state.prompt_tokens_len = len(state.tokens)

                # 保存请求参数
                parameters = json.loads(request.parameters())
                state.ignore_eos = parameters["ignore_eos"]
                state.max_tokens = parameters["max_tokens"]

                self.state[correlation_id] = state

            # 非首次调度：直接取该请求已累计的 token 序列
            input_ids.append(self.state[correlation_id].tokens)

        # 找到批内最长序列，其余序列左侧用 EOS 填充对齐
        max_len = max([len(x) for x in input_ids])

        # 填充输入张量：左侧补 EOS token
        input_ids_torch = torch.cat(
            [
                torch.cat(
                    [
                        torch.tensor(
                            [self.tokenizer.eos_token_id] * (max_len - len(x)),
                            device=self.device,
                        )
                    ]
                    + [x]
                ).unsqueeze(0)
                for x in input_ids
            ]
        )
        # 对应的 attention mask：填充位为 0，真实 token 为 1
        attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.tensor([0] * (max_len - x.numel())),
                        torch.tensor([1] * x.numel()),
                    ]
                ).unsqueeze(0)
                for x in input_ids
            ]
        )
        return input_ids_torch.long(), attention_mask.long().to(self.device)

    # 把生成的 token 逐条发回客户端；未完成的请求标记为重新调度（reschedule）
    def send_responses(self, requests, outputs):
        """
        Scatter method for processing requests and sending responses.

        Args:
            requests (list): List of Triton InferenceRequest objects.
            outputs (list): List of output tensors generated by the model.

        Returns:
            None
        """
        for i, request in enumerate(requests):
            correlation_id = (
                pb_utils.get_input_tensor_by_name(request, "correlation_id")
                .as_numpy()
                .item()
            )
            # 获取该请求的响应发送器（decoupled 模式下逐 token 流式发送）
            response_sender = request.get_response_sender()
            # 把标量 token 转成一维张量
            generated_token = outputs[i][-1].reshape(1)

            ignore_eos = self.state[correlation_id].ignore_eos

            # 生成长度上限 = 用户指定 max_tokens + 提示词长度
            max_tokens = (
                self.state[correlation_id].max_tokens
                + self.state[correlation_id].prompt_tokens_len
            )

            # 把新生成的 token 追加到该请求的状态序列
            self.state[correlation_id].tokens = torch.cat(
                [self.state[correlation_id].tokens, generated_token]
            )
            # 生成了 EOS（且未忽略）或达到长度上限 → 发送最终响应并释放请求
            if (
                generated_token.item() == self.tokenizer.eos_token_id and not ignore_eos
            ) or len(self.state[correlation_id].tokens) >= max_tokens:
                flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                request.set_release_flags(pb_utils.TRITONSERVER_REQUEST_RELEASE_ALL)
                del self.state[correlation_id]
            # 否则：标记为 RESCHEDULE，请求会带着已生成 token 被重新调度（迭代调度核心）
            else:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                flags = 0

            output_decoded = self.tokenizer.decode(generated_token.cpu().item())
            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "text_output", np.array([output_decoded], dtype=np.object_)
                    )
                ]
            )
            response_sender.send(response, flags=flags)

    # 每次调度：只生成 1 个新 token；新到达的请求会与"飞行中"的请求合批
    def execute(self, requests):
        pb_utils.Logger.log_verbose(f"Processing {len(requests)} request(s).")
        input_ids, attention_mask = self.create_batch(requests)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        self.send_responses(requests, outputs)
