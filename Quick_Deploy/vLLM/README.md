<!--
# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# 在 Triton 中部署 vLLM 模型

本教程演示如何借助 Triton 基于 Python 的 [vLLM](https://github.com/triton-inference-server/vllm_backend/tree/main) 后端，在 Triton Inference Server 上部署一个简单的 [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) 模型。

*注意*：本教程仅作为参考示例，存在[已知限制](#limitations)。

## 第一步：准备模型仓库

使用 Triton 前，需要先构建模型仓库（model repository）。本教程将直接使用 [vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main) 仓库 [samples](https://github.com/triton-inference-server/vllm_backend/tree/main/samples) 目录中提供的模型仓库。

下面这组命令会创建 `model_repository/vllm_model/1` 目录，并复制两份服务 [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) 模型所必需的文件：[`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json) 和 [`config.pbtxt`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/config.pbtxt)。

```
mkdir -p model_repository/vllm_model/1
wget -P model_repository/vllm_model/1 https://raw.githubusercontent.com/triton-inference-server/vllm_backend/r<xx.yy>/samples/model_repository/vllm_model/1/model.json
wget -P model_repository/vllm_model/ https://raw.githubusercontent.com/triton-inference-server/vllm_backend/r<xx.yy>/samples/model_repository/vllm_model/config.pbtxt
```

其中 `<xx.yy>` 是你要使用的 Triton 版本号。请注意，Triton 的 vLLM 容器自 23.10 版本起才提供。

模型仓库最终应呈如下结构：
```
model_repository/
└── vllm_model
    ├── 1
    │   └── model.json
    └── config.pbtxt
```

`model.json` 的内容为：

```json
{
    "model": "facebook/opt-125m",
    "gpu_memory_utilization": 0.5
}
```

可以修改该文件，为 vLLM 引擎提供更多设置项。vLLM 支持的键值对请参见 [AsyncEngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L165) 和 [EngineArgs](https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/arg_utils.py#L11)。飞行批处理（inflight batching）和分页注意力（paged attention）由 vLLM 引擎负责处理。

> 💡 **AI Infra 视角**：与上一节部署 CNN 图像模型不同，LLM 推理是自回归的——每个请求需要逐 token 生成，且中间要缓存 KV 缓存（KV cache）来避免重复计算。vLLM 通过分页注意力将 KV 缓存按页管理、按需分配，再用飞行批处理让不同请求交替占用 GPU 计算资源，从而大幅提升吞吐。`gpu_memory_utilization` 决定 vLLM 预留多少显存用于 KV 缓存：设得太高会与模型权重争抢显存导致 OOM，设得太低则会浪费缓存空间、降低并发能力，生产环境通常需要结合压测调优。

如需多 GPU 支持，可以在 [`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json) 中指定 `tensor_parallel_size` 之类的 EngineArgs 参数。

*注意*：vLLM 在默认设置下会贪婪地占用 GPU 高达 90% 的显存。本教程通过将 `gpu_memory_utilization` 设为 50% 来调整这一行为。你可以通过 [`model.json`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json) 中的 `gpu_memory_utilization` 等字段来微调该行为。

> 💡 **AI Infra 视角**：`tensor_parallel_size` 用于把模型切分到多张 GPU 上（张量并行），是单卡装不下大模型（如 70B 及以上参数量）时的标准解法。代价是每层前向传播都需要跨卡通信，因此在多机多卡场景下，网络带宽（NVLink/InfiniBand）往往成为吞吐瓶颈，选型时需一并考虑。

请阅读 [`model.py`](https://github.com/triton-inference-server/vllm_backend/blob/main/src/model.py) 中的文档，了解如何针对你的使用场景配置这个示例。

## 第二步：启动 Triton Inference Server

模型仓库准备就绪后，就可以启动 Triton 服务器了。自 23.10 版本起，NGC 上提供了预装 vLLM 的专用容器。要使用该容器启动 Triton，可以执行下面的 docker 命令。

```
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 tritonserver --model-store ./model_repository
```

整个教程中，\<xx.yy\> 是你要使用的 Triton 版本号。请注意，Triton 的 vLLM 容器最早在 23.10 版本发布，更早的版本均无法使用。

Triton 启动后，控制台会输出服务器启动和加载模型的信息。当看到类似下面的输出时，说明 Triton 已就绪，可以接受推理请求了。

```
I1030 22:33:28.291908 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1030 22:33:28.292879 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1030 22:33:28.335154 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

## 第三步：用 Triton 客户端发送第一个推理请求

本教程将演示两种向 [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) 模型发送推理请求的方式：

* [使用 generate 端点](#using-the-generate-endpoint)
* [使用 gRPC asyncio 客户端](#using-the-grpc-asyncio-client)

### 使用 generate 端点
用示例模型仓库启动 Triton 后，你可以通过 [generate](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md) 端点快速发起第一个推理请求。

先用下面的命令启动 Triton 的 SDK 容器：
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk bash
```

现在，发送一个推理请求：
```
curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
```

成功后会收到类似如下的服务器响应：
```
{"model_name":"vllm_model","model_version":"1","text_output":"What is Triton Inference Server?\n\nTriton Inference Server is a server that is used by many"}
```

### 使用 gRPC asyncio 客户端
下面演示如何在 Triton 的 SDK 容器内，借助 [gRPC asyncio 客户端](https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/aio/__init__.py) 库发起多个异步请求。

> 💡 **AI Infra 视角**：传统 CNN 推理是"一次请求、一次响应"的同步往返；而 LLM 推理是流式的——文本逐 token 生成，首 token 延迟（TTFT）与整体吞吐（TPS）是两种不同的优化目标。Triton 的 generate 端点正是为 LLM 这类场景设计的简化接口，返回文本而非张量；而异步客户端则用于高并发场景，让一个进程同时维护大量在途请求，避免因等待响应而阻塞。

这种方式需要 [client.py](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/client.py) 脚本和一组 [prompts](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/prompts.txt)，两者都在 [vllm_backend](https://github.com/triton-inference-server/vllm_backend/tree/main) 仓库的 [samples](https://github.com/triton-inference-server/vllm_backend/tree/main/samples) 目录中。

使用下面的命令把 `client.py` 和 `prompts.txt` 下载到当前目录：
```
wget https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/samples/client.py
wget https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/samples/prompts.txt
```

现在，可以启动 Triton 的 SDK 容器了：
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk bash
```

在容器内运行 [`client.py`](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/client.py)：
```
python3 client.py
```

客户端从 [prompts.txt](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/prompts.txt) 文件读取提示词，发送给 Triton 服务器进行推理，并将结果默认保存到名为 `results.txt` 的文件中。

客户端输出大致如下：

```
Loading inputs from `prompts.txt`...
Storing results into `results.txt`...
PASS: vLLM example
```

你可以查看 `results.txt` 的内容来了解服务器的响应。客户端还支持 `--iterations` 参数，通过循环遍历 [prompts.txt](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/prompts.txt) 中的提示词列表来增大服务器负载。

以 `--verbose` 参数运行客户端时，会打印更多关于请求/响应事务的细节。

## 限制（Limitations）

- 即使每个请求只有 1 个响应，我们也使用解耦（decoupled）流式协议。
- asyncio 实现暴露给了 model.py。
- 不支持指定使用哪一部分 GPU。
- 如果同时运行多个 Triton 服务器实例，且都使用基于 Python 的 vLLM 后端，则需要为每个服务器指定不同的 `shm-region-prefix-name`。更多信息请参见[这里](https://github.com/triton-inference-server/python_backend#running-multiple-instances-of-triton-server)。
