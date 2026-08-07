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

# 使用 Triton 与 vLLM 部署 Llama2-7B 模型

vLLM Backend 使用 vLLM 执行推理。关于 vLLM 的更多信息请参阅[这里](https://blog.vllm.ai/2023/06/20/vllm.html)，
关于 vLLM Backend 请参阅[这里](https://github.com/triton-inference-server/vllm_backend)。

> 💡 **AI Infra 视角**：与 TRT-LLM 相比，vLLM 的最大优势是"零编译"：直接从 HuggingFace 权重启动，无需构建引擎，换模型、换版本的成本极低，因此特别适合模型频繁迭代或需要快速上线多个模型的场景。代价是通用实现相比 TRT-LLM 的针对性编译，在极端延迟和峰值吞吐上略逊一筹。选型建议：追求极致性能且模型稳定选 TRT-LLM，追求迭代速度和多模型灵活切换选 vLLM。

## 预构建说明

本教程使用带预训练权重的 Llama2-7B HuggingFace 模型。请遵循 [README.md](README.md) 中的预构建说明，并获取在其他后端上运行 Llama 的链接。

## 安装

Triton vLLM 容器可以从 [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) 拉取：

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v $PWD/llama2vllm:/opt/tritonserver/model_repository/llama2vllm \
    nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3
```
这会创建一个 `/opt/tritonserver/model_repository` 文件夹，内含 `llama2vllm` 模型。模型本身会从 HuggingFace 拉取。

> 💡 **AI Infra 视角**：这里模型权重是服务启动时从 HuggingFace 在线拉取的，适合演示；生产环境不建议这么做——首次拉取可能耗时数分钟、依赖外网可用性，且模型仓库版本可能悄悄变化。规范做法是把权重预先下载到内部存储（对象存储或共享卷），通过挂载或离线镜像分发到 GPU 节点，让容器启动路径完全离线、版本可锁定、可回滚。

进入容器后，安装 `huggingface-cli` 并用你自己的凭据登录。
```bash
pip install --upgrade huggingface_hub
huggingface-cli login --token <your huggingface access token>
```


## 用 Triton 提供服务

然后按常规方式运行 tritonserver：
```bash
tritonserver --model-repository model_repository
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
$ curl -X POST localhost:8000/v2/models/llama2vllm/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
# returns (formatted for better visualization)
> {
    "model_name":"llama2vllm",
    "model_version":"1",
    "text_output":"What is Triton Inference Server?\nTriton Inference Server is a lightweight, high-performance"
  }
```

## 通过 Triton client 发送请求

Triton vLLM Backend 仓库有一个 [samples 目录](https://github.com/triton-inference-server/vllm_backend/tree/main/samples)，
里面提供了用于测试 Llama2 模型的示例 client.py。

```bash
pip3 install tritonclient[all]
# Assuming Tritonserver server is running already
$ git clone https://github.com/triton-inference-server/vllm_backend.git
$ cd vllm_backend/samples
$ python3 client.py -m llama2vllm

```
执行上述步骤后，会生成一个内容如下的 `results.txt`
```bash
Hello, my name is
I am a 20 year old student from the Netherlands. I am currently

=========

The most dangerous animal is
The most dangerous animal is the one that is not there.
The most dangerous

=========

The capital of France is
The capital of France is Paris.
The capital of France is Paris. The

=========

The future of AI is
The future of AI is in the hands of the people who use it.

=========
```
