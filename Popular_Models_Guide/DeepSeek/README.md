<!--
# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->
# 使用 Triton 部署 DeepSeek-R1-Distill-Llama-8B 模型

本教程将使用 vLLM Backend 部署
[`DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)。
关于 vLLM 的更多信息请参阅[这里](https://blog.vllm.ai/2023/06/20/vllm.html)，
关于 vLLM Backend 请参阅[这里](https://github.com/triton-inference-server/vllm_backend)。

> 💡 **AI Infra 视角**：vLLM 的核心创新是持续批处理（continuous batching）与 PagedAttention：传统批处理必须等整批请求都完成才释放资源，而持续批处理允许请求按 token 粒度动态进出批次，GPU 上随时都跑着进度各异的请求，大幅提升吞吐。PagedAttention 则把 KV cache 按页管理，像操作系统管理内存一样按需分配，消除了显存碎片。这也是 vLLM 成为当前 LLM 服务主流引擎的原因。

## 模型仓库

首先搭建一个模型仓库。本教程使用 [Triton vLLM backend 仓库](https://github.com/triton-inference-server/vllm_backend/tree/main/samples/model_repository/vllm_model) 中提供的示例模型仓库。

克隆完整仓库：
```bash
git clone -b r25.01 https://github.com/triton-inference-server/vllm_backend.git
```

示例模型仓库使用的是 [`facebook/opt-125m` 模型](https://github.com/triton-inference-server/vllm_backend/blob/80dd0371e0301fabf79c57536e60700d016fcc76/samples/model_repository/vllm_model/1/model.json#L2)，
我们把它替换成 `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"`。另外请注意，使用默认参数时，需要根据你的硬件情况适当调整 `gpu_memory_utilization`。在全部使用默认参数的情况下，
通过 Triton + vLLM backend 部署 `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"` 大约需要 35GB 显存，请据此调整 `gpu_memory_utilization`。
例如，RTX 5880 上最小值应为 `0.69`，而 A100 上 `0.41` 就足够了。为了本教程简单起见，我们把它设为 `0.9`。修改后的 `model.json` 如下：
```json
{
    "model":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "gpu_memory_utilization": 0.9,
    "enforce_eager": true
}
```

> 💡 **AI Infra 视角**：`gpu_memory_utilization` 决定 vLLM 为模型预留的显存比例。大模型推理的显存占用主要有三块：模型权重（weights）、KV cache（为每个正在处理的请求缓存的注意力键值）和激活值（activations）。前两者是占大头且相对可预算的——权重是固定大小，KV cache 随并发请求数增长。调高该值意味着给 KV cache 留更多空间，能支撑更高并发，但挤占了其他用途的显存，调过头会导致申请显存失败或与别的模型打架。做容量规划时，先用 `nvidia-smi` 观察空闲显存，再按并发需求反推这个值。

## 用 Triton 提供服务

然后按常规方式启动 tritonserver
```bash
LOCAL_MODEL_REPOSITORY=./vllm_backend/samples/model_repository/
docker run --rm -it --net host --shm-size=2g  --ulimit memlock=-1 \
--ulimit stack=67108864 --gpus all -v $LOCAL_MODEL_REPOSITORY:/opt/tritonserver/model_repository  \
nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3 tritonserver --model-repository=model_repository/
```
当控制台出现以下输出时，说明服务器启动成功：

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## 通过 `generate` 端点发送请求

作为一个简单的验证示例，你可以用 `generate` 端点测试服务器是否正常工作。关于 generate 端点的更多信息请参阅[这里](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)。

```bash
$ curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "exclude_input_in_output": true, "max_tokens": 45}}' | jq
```
预期输出如下：
```json
{
  "model_name": "vllm_model",
  "model_version": "1",
  "text_output": " It's a high-performance, scalable, and efficient inference server for AI models. It's designed to handle large numbers of requests quickly and efficiently, making it suitable for real-time applications like autonomous vehicles, smart homes, and more"
}
```

> 💡 **AI Infra 视角**：`generate` 端点是 Triton 为生成式 AI 模型提供的专用 HTTP 扩展接口，把"拼接提示词、调用模型、解码输出"打包成一个简单调用，免去手工拼装张量（tensor）的麻烦。它支持流式返回（`stream: true`）和各类采样参数，适合快速联调。生产接入则更多走标准的 gRPC/HTTP 推理协议，以获得更强的类型检查和性能控制。

## 通过 Triton client 发送请求

Triton vLLM Backend 仓库有一个 [samples 目录](https://github.com/triton-inference-server/vllm_backend/tree/main/samples)，
里面提供了用于测试模型的示例 client.py。

```bash
LOCAL_WORKSPACE=./vllm_backend/samples
docker run -ti --gpus all --network=host --pid=host --ipc=host -v $LOCAL_WORKSPACE:/workspace nvcr.io/nvidia/tritonserver:26.07-py3-sdk
```
然后按如下方式使用客户端：
```bash
python client.py -m vllm_model
```

执行上述步骤后，会生成一个内容如下的 `results.txt`
```
Hello, my name is
I need to write a program that can read a text file and find all the names in the text. The names can be in any case (uppercase, lowercase, or mixed). Also, the names can be part of longer words or phrases, so I need to make sure that I'm extracting only the names and not parts of other words. Additionally, the names can be separated by various non-word characters, such as commas, periods, apostrophes, etc. So, I need to extract

=========

The most dangerous animal is
The most dangerous animal is the one that poses the greatest threat to human safety and well-being. This can vary depending on the region and the specific circumstances. For example, in some areas, large predators like lions or tigers might be considered the most dangerous, while in others, venomous snakes or dangerous marine animals might take precedence.

To determine the most dangerous animal, one would need to consider factors such as:
1. **Number of incidents**: How many people have been injured or killed by this

=========

The capital of France is
A) London
B) Paris
C) Marseille
D) Lyon

Okay, so I have this question here: "The capital of France is..." with options A) London, B) Paris, C) Marseille, D) Lyon. Hmm, I need to figure out the correct answer. Let me think about what I know regarding the capitals of different countries.

First off, I remember that France is a country in Western Europe. I've heard people talk about Paris before, especially in

=========

The future of AI is
AI is the future of everything. It's going to change how we live, work, and interact with the world. From healthcare to education, from transportation to entertainment, AI will play a crucial role in shaping our tomorrow. But what does that mean for us? How will AI impact our daily lives? Let's explore some possibilities.

First, in healthcare, AI can help diagnose diseases faster and more accurately than ever before. It can analyze medical data, recommend treatments, and even assist in surgery.

=========
```
