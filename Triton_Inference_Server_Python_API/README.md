<!--
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton 推理服务器进程内 Python API（In-Process Python API）[BETA]

从 24.01 版本开始，Triton Inference Server 附带一个 Python 包，开发者可以借此把 Triton Inference Server 实例直接嵌入到自己的 Python 应用中。进程内 Python API 在设计上对齐进程内 C API（in-process C API）的功能，同时提供更高层的抽象。其核心是 C API 的一对一 Python 绑定，既保留了 C API 的全部灵活性与能力，又提供了更简单的使用接口。

> [!Note]
> 该 API 仍处于 BETA 阶段，我们在测试各种特性并收集反馈的过程中，接口可能会有调整。
> 我们欢迎一切反馈，期待听到你的声音！

| [环境要求](#环境要求) | [安装](#安装) | [Hello World](#hello-world) | [Stable Diffusion](#stable-diffusion) | [Ray Serve 部署](examples/rayserve) |

## 环境要求

以下操作需要一台装有 Docker 的 Linux 系统。如需 CUDA 支持，请确认你的 CUDA 驱动满足深度学习框架支持矩阵中「NVIDIA Driver」一节的要求：
https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

## 安装

本教程与 Python API 包设计为在 `nvcr.io/nvidia/tritonserver:26.07-py3` Docker 镜像内安装和运行。

我们提供了一组便捷脚本，用于基于 `nvcr.io/nvidia/tritonserver:26.07-py3` 镜像创建 Docker 镜像，其中已装好 Python API 以及示例所需的额外依赖。

### Triton Inference Server 26.07 + Python API

#### 克隆仓库
```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Triton_Inference_Server_Python_API
```

#### 构建 `triton-python-api:r26.07` 镜像
```bash
./build.sh
```

#### 支持的 Backend

构建出的镜像包含 tritonserver `nvcr.io/nvidia/tritonserver:26.07-py3` 容器默认随附的全部 backend。

```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### 内置模型

`default` 构建包含一个 `identity` 模型，可用于演练基本操作，包括发送不同数据类型的输入张量。`identity` 模型会把形状为 `[-1, -1]` 的输入原样复制到形状为 `[-1, -1]` 的输出。输入名称为 `data_type_input`，输出名称为 `data_type_output`（例如 `string_input`、`string_output`、`fp16_input`、`fp16_output`）。


## Hello World

### 启动 `triton-python-api:r26.07` 容器

下面的命令会启动一个容器，并把当前目录挂载为 `workspace`。

```bash
./run.sh
```

### 进入 Python Shell

```bash
python3
```

### 创建并启动服务器实例

> 💡 **AI Infra 视角**：进程内 Python API 把 Triton 当作一个库嵌入 Python 应用，而不是独立起一个服务进程、再通过 HTTP/gRPC 访问。这样省去了网络序列化开销，也免去部署两个组件，适合在 Python 编排层内部直接调用推理。与之相对，独立部署形态（standalone）仍是生产环境的主流，因为它便于跨语言调用、独立扩缩容和故障隔离——选择哪种形态，本质上是在「嵌入的灵活度」和「部署的解耦度」之间权衡。

```python
import tritonserver

server = tritonserver.Server(model_repository="/workspace/identity-models")
server.start()
```

### 列出模型

```
server.models()
```

#### 示例输出

`server.models()` 返回一个字典，记录可用模型及其当前状态。

```python
{('identity', 1): {'name': 'identity', 'version': 1, 'state': 'READY'}}
```

### 发送推理请求

```python
model = server.model("identity")
responses = model.infer(inputs={"string_input":[["hello world!"]]})
```

### 遍历响应

`model.infer()` 返回一个迭代器，可用于逐个处理推理请求的结果。

> 💡 **AI Infra 视角**：`model.infer()` 返回迭代器而不是单个结果，对应 Triton 的异步推理（asynchronous inference）模型：请求发出后不必阻塞等待，结果可以流式逐个消费。这种设计是解耦（decoupled）模型与流式推理（streaming，如 LLM 逐 token 输出）的基础；在生产中，异步推理配合批处理（batching）是榨干 GPU 利用率的关键手段。

```python
for response in responses:
    print(response.outputs["string_output"].to_string_array())
```

#### 示例输出
```python
[['hello world!']]
```


## Stable Diffusion

本示例基于 [Popular_Models_Guide/StableDiffusion](../Popular_Models_Guide/StableDiffusion) 教程。

> 💡 **AI Infra 视角**：Stable Diffusion 这类生成式管线通常由多个模型串成一条流水线（文本编码器、UNet、VAE 解码器），在 Triton 中既可把每个环节做成独立模型、由 ensemble 编排，也可整体作为单个 Python backend 模型运行。后者把「一个推理请求 = 一整张图」作为对外粒度，屏蔽内部多阶段细节，正是进程内 API 最常见的落地方式之一。

#### 构建 `triton-python-api:r26.07-diffusion` 镜像与 Stable Diffusion 模型

请注意，以下命令可能需要数分钟甚至更久，具体取决于你的硬件配置和网络连接。

```bash
   ./build.sh --framework diffusion --build-models
```

#### 支持的 Backend

构建出的镜像包含 tritonserver `nvcr.io/nvidia/tritonserver:26.07-py3` 容器默认随附的全部 backend。

```
dali  fil  identity  onnxruntime  openvino  python  pytorch  repeat  square  tensorflow  tensorrt
```

#### 内置模型

`diffusion` 构建包含一个 `stable_diffustion` 管线，输入文本提示词（text prompt），返回生成的图片。关于模型与管线的更多细节，请参阅 [Popular_Models_Guide/StableDiffusion](../Popular_Models_Guide/StableDiffusion) 教程。

### 启动容器

下面的命令会启动一个容器，并把当前目录挂载为 `workspace`。

```bash
./run.sh --framework diffusion
```

### 进入 Python Shell

```bash
python3
```

### 创建并启动服务器实例

```python
import tritonserver
import numpy
from PIL import Image

server = tritonserver.Server(model_repository="/workspace/diffusion-models")
server.start()
```

### 列出模型

```
server.models()
```

#### 示例输出
```python
{('stable_diffusion_1_5', 1): {'name': 'stable_diffusion_1_5', 'version': 1, 'state': 'READY'}, ('stable_diffusion_xl', 1): {'name': 'stable_diffusion_xl', 'version': 1, 'state': 'READY'}}
```

### 发送推理请求

```python
model = server.model("stable_diffusion_xl")
responses = model.infer(inputs={"prompt":[["butterfly in new york, realistic, 4k, photograph"]]})
```

### 遍历响应并保存图片


```python
for response in responses:
	generated_image = numpy.from_dlpack(response.outputs["generated_image"])
	generated_image = generated_image.squeeze().astype(numpy.uint8)
	image_ = Image.fromarray(generated_image)
	image_.save("sample_generated_image.jpg")
```

#### 示例输出

![sample_generated_image](./docs/sample_generated_image.jpg)
