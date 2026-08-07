<!--
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
-->

# 使用 Python Backend 和迭代调度部署 GPT-2 模型

| 跳转到 | [第 6 部分：构建复杂流水线：Stable Diffusion](../Part_6-building_complex_pipelines)  | [文档：迭代调度（Iterative Scheduling）](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#iterative-sequences) |
| ------------ | --------------- | --------------- |

在本教程中，我们将使用 Python backend 部署一个 GPT-2 模型，并演示[迭代调度（iterative scheduling）](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#iterative-sequences)特性。

## 前置条件

开始本教程之前，请确保你熟悉以下概念：

* [Triton-Server 快速入门](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
* [Python Backend](https://github.com/triton-inference-server/python_backend)

## 迭代调度（Iterative Scheduling）

迭代调度是一种让 Triton Inference Server 用相同输入多次调度同一条请求的技术。这对具有自回归循环（auto-regressive loop）的模型很有用。迭代调度让 Triton Server 能为你的模型实现飞行中批处理（inflight batching），并让你可以把新到达的序列与飞行中的序列合并在一起。

> 💡 **AI Infra 视角**：这里说的 inflight batching 正是 LLM 推理中 continuous batching（连续批处理）的思想雏形：LLM 逐 token 生成时，每个请求的生成进度不同，如果强制整批同步推进，先完成的请求只能空等。迭代调度允许"边生成边插新请求"，让 GPU 始终有活干——这是 LLM 服务吞吐的关键机制，vLLM 等引擎也采用同样的思路。

## 教程概览

本教程部署两个模型：

* simple-gpt2：该模型接收一批请求，只有在完成当前批次的 token 生成后才进入下一个批次。

* iterative-gpt2：该模型使用迭代调度，即使仍在为之前的序列生成 token，也能处理批次中的新序列。

### 演示

[![asciicast](https://asciinema.org/a/TUZtHwZsYrJzHuZF7XCOj1Avx.svg)](https://asciinema.org/a/TUZtHwZsYrJzHuZF7XCOj1Avx)

### 第 1 步：准备服务器环境

* 首先，运行 Triton Inference Server 容器：

```
# Replace yy.mm with year and month of release. Please use 24.04 release upward.
docker run --gpus=all --name iterative-scheduling -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```

* 接下来，安装 python backend 中模型运行所需的所有依赖，并用你的 [huggingface token](https://huggingface.co/settings/tokens) 登录（需要有 [HuggingFace](https://huggingface.co/) 账号）。

```
pip install transformers[torch]
```

> [!NOTE]
> 可选：如果不想每次运行容器时都重新安装依赖，可以运行 `docker commit iterative-scheduling iterative-scheduling-image` 保存容器，后续运行直接使用保存的镜像。

然后启动服务器：

```
tritonserver --model-repository=/models
```

### 第 2 步：安装客户端依赖

在另一个终端中安装客户端依赖：

```
pip3 install tritonclient[grpc]
pip3 install tqdm
```

### 第 3 步：运行客户端

simple-gpt2 模型不使用迭代调度，只有在完成当前批次的 token 生成后才会进入下一个批次。

运行以下命令启动客户端：

```
python3 client/client.py --model simple-gpt2
```

可以看到，一个请求的 token 先生成完毕，然后才轮到下一个请求。

按 `Ctrl+C` 停止客户端。

迭代调度器则能够把到达服务器的新请求合并进来。

运行以下命令启动客户端：

```
python3 client/client.py --model iterative-gpt2
```

可以看到，两个提示词的 token 在同时生成。

## 后续步骤

我们计划为这些模型集成 KV-Cache 以获得更好的性能。目前，本教程的主要目标是演示如何将迭代调度与 Python backend 配合使用。
