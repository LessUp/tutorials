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

# 使用 Triton 与 TensorRT 部署 Stable Diffusion 模型

本示例演示如何借助 [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/10.4/demo/Diffusion)
的管线与工具，在 Triton 中部署 Stable Diffusion 模型。

在 TensorRT demo 的基础上，本示例提供了一个可复用的
[Python 后端](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md)（即 [`/backend/diffusion/model.py`](backend/diffusion/model.py)），
适合部署多种版本、多种配置的 Diffusion 模型。

> 💡 **AI Infra 视角**：Python 后端（Python backend）是 Triton 中最灵活的一类后端，推理逻辑完全由用户用 Python 编写，适合 TensorRT 引擎、ONNX 等标准格式无法覆盖的定制模型（比如本例的扩散模型管线）。它的代价是相比原生 C++ 后端多了 Python 解释和序列化开销，生产中若吞吐不达标，通常的做法是先用 Python 后端跑通业务，再针对热点路径做原生化改造。

关于 Stable Diffusion 的更多信息请参考
[stable-diffusion-v1-5](https://huggingface.co/benjamin-paine/stable-diffusion-v1-5) 与
[stable-diffusion-xl](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)。关于
TensorRT 实现的更多信息请参考 [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/10.4/demo/Diffusion)。

> [!Note]
> 本示例仅作为样例代码提供，用于生产环境前请务必仔细审查。

| [环境要求](#requirements) | [构建 Triton Inference Server 镜像](#building-the-triton-inference-server-image) | [Stable Diffusion v1.5](#building-and-running-stable-diffusion-v-15) | [Stable Diffusion XL](#building-and-running-stable-diffusion-xl) | [发送推理请求](#sending-an-inference-request) | [模型配置](docs/model_configuration.md) | [示例客户端](#sample-client) | [已知问题与限制](#known-issues-and-limitations) |

## 环境要求

以下操作需要一台安装有 Docker 的 Linux 系统。若要启用 CUDA，请确保你的 CUDA 驱动满足
[深度学习框架支持矩阵](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)中
"NVIDIA Driver" 一节的要求。

## 构建 Triton Inference Server 镜像

本示例基于 `nvcr.io/nvidia/tritonserver:26.07-py3` Docker 镜像和
[TensorRT OSS v10.4](https://github.com/NVIDIA/TensorRT/releases/tag/v10.4.0) 设计。

项目提供了一组便捷脚本，用于在 `nvcr.io/nvidia/tritonserver:26.07-py3` 镜像的基础上
构建一个安装了 TensorRT Stable Diffusion demo 所需依赖的 Docker 镜像。

### Triton Inference Server + TensorRT OSS

#### 克隆仓库
```bash
git clone https://github.com/triton-inference-server/tutorials.git --single-branch
cd tutorials/Popular_Models_Guide/StableDiffusion
```

#### 构建 Tritonserver Diffusion Docker 镜像
```bash
./build.sh
```

#### 内置模型

`default` 构建包含位于 `/diffusion-models` 目录下的模型配置文件。示例配置为
[`stable_diffusion_1_5`](diffusion-models/stable_diffusion_1_5) 和
[`stable_diffusion_xl`](diffusion-models/stable_diffusion_xl)。

模型产物（artifacts）与引擎文件不包含在镜像中，需要单独构建到一个挂载卷目录里。

## 构建并运行 Stable Diffusion v 1.5

### 启动 Tritonserver Diffusion 容器

下面的命令会启动一个容器，并把当前目录以 `workspace` 挂载进容器。

```bash
./run.sh
```

### 构建 Stable Diffusion v 1.5 引擎

> [!Note]
>
> 模型
> [stable-diffusion-v1-5](https://huggingface.co/benjamin-paine/stable-diffusion-v1-5)
> 需要登录 HuggingFace 并接受其使用条款。请相应设置环境变量 HF_TOKEN。
>

```bash
./scripts/build_models.sh --model stable_diffusion_1_5
```

#### 预期输出
```
 diffusion-models
|-- stable_diffusion_1_5
|   |-- 1
|   |   |-- 1.5-engine-batch-size-1
|   |   |-- 1.5-onnx
|   |   |-- 1.5-pytorch_model
|   `-- config.pbtxt

```

> 💡 **AI Infra 视角**：这里的 "engine"（引擎）是把 PyTorch 模型经过 ONNX 转换后，再编译成针对当前 GPU 硬件与指定批大小（batch size）高度优化的 TensorRT 执行计划（execution plan）。引擎构建可能耗时数分钟到数十分钟，但推理时性能远优于原始 PyTorch——这就是"编译期优化换运行时性能"的典型 trade-off。注意引擎与硬件型号、批大小强绑定，换 GPU 或改 batch size 都要重新构建，这也是生产里常把引擎构建流程单独 CI 化的原因。

### 启动服务器实例

> [!Note]
> 为了演示方便，我们使用 `EXPLICIT` 模型控制模式来精确控制加载哪个 stable diffusion 版本。生产部署请参考 [安全部署注意事项][secure_guide]
> 了解 `EXPLICIT` 模式相关的风险。

[secure_guide]: https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/deploy.md

```bash
tritonserver --model-repository diffusion-models --model-control-mode explicit --load-model stable_diffusion_1_5
```

#### 预期输出
```
<SNIP>
I0229 20:15:52.125050 749 server.cc:676]
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| stable_diffusion_1_5 | 1       | READY  |
+----------------------+---------+--------+

<SNIP>
```

> 💡 **AI Infra 视角**：模型控制模式（model control mode）决定 Triton 何时加载/卸载模型。默认的 `NONE` 模式是启动时自动加载仓库里的全部模型；`EXPLICIT` 模式下模型必须通过显式请求（如上例的 `--load-model`）才会加载。`EXPLICIT` 的好处是能精确控制显存占用、避免无关模型抢占资源，坏处是配置失误会导致模型缺失、服务不可用——生产环境常用它配合 K8s 做按需加载。

## 构建并运行 Stable Diffusion XL

### 启动 Tritonserver Diffusion 容器

下面的命令会启动一个容器，并把当前目录以 `workspace` 挂载进容器。

```bash
./run.sh
```

### 构建 Stable Diffusion XL 引擎

```bash
./scripts/build_models.sh --model stable_diffusion_xl
```

#### 预期输出
```
 diffusion-models
 |-- stable_diffusion_xl
    |-- 1
    |   |-- xl-1.0-engine-batch-size-1
    |   |-- xl-1.0-onnx
    |   `-- xl-1.0-pytorch_model
    `-- config.pbtxt
```

### 启动服务器实例

> [!Note]
> 为了演示方便，我们使用 `EXPLICIT` 模型控制模式来精确控制加载哪个 stable diffusion 版本。生产部署请参考 [安全部署注意事项][secure_guide]
> 了解 `EXPLICIT` 模式相关的风险。


```bash
tritonserver --model-repository diffusion-models --model-control-mode explicit --load-model stable_diffusion_xl
```

#### 预期输出
```
<SNIP>
I0229 20:22:22.912465 1440 server.cc:676]
+---------------------+---------+--------+
| Model               | Version | Status |
+---------------------+---------+--------+
| stable_diffusion_xl | 1       | READY  |
+---------------------+---------+--------+

<SNIP>
```

## 发送推理请求

我们提供了一个示例 [客户端](client.py) 程序，让发送和接收请求更简单。

### 启动 Tritonserver Diffusion 容器

在运行服务器的另一个终端里启动一个新容器。

下面的命令会启动一个容器，并把当前目录以 `workspace` 挂载进容器。

```bash
./run.sh
```


### 向 Stable Diffusion 1.5 发送提示词

```bash
python3 client.py --model stable_diffusion_1_5 --prompt "butterfly in new york, 4k, realistic" --save-image
```

#### 示例输出

```bash
Client: 0 Throughput: 0.7201335361144658 Avg. Latency: 1.3677194118499756
Throughput: 0.7163933558221957 Total Time: 1.395881175994873
```

如果指定了 `--save-image`，生成的图片会保存为 jpeg 文件。

`
 client_0_generated_image_0.jpg
`

![sample_generated_image](./docs/client_0_generated_image_0_1_5.jpg)


### 向 Stable Diffusion XL 发送提示词

```bash
python3 client.py --model stable_diffusion_xl --prompt "butterfly in new york, 4k, realistic" --save-image
```

#### 示例输出

```bash
Client: 0 Throughput: 0.1825067711674996 Avg. Latency: 5.465569257736206
Throughput: 0.18224859609447058 Total Time: 5.487010717391968
```

如果指定了 `--save-image`，生成的图片会保存为 jpeg 文件。

`
 client_0_generated_image_0.jpg
`

![sample_generated_image](./docs/client_0_generated_image_0_xl.jpg)


## 示例客户端

示例 [客户端](client.py) 程序可以让用户在不同并发场景下快速测试
diffusion 模型。查看客户端程序全部选项及说明请运行：

```
python3 client.py --help
```

### 发送并发请求

要增加负载和并发度，可以用 `clients` 和
`requests` 选项分别控制客户端进程数和每个客户端发送的请求数。

#### 示例：十个客户端各发送十个请求

下面的命令会启动十个客户端，每个客户端发送十个
请求。每个客户端都是独立的进程，并行地与其他九个客户端逐个发送请求。

```bash
python3 client.py --model stable_diffusion_xl --requests 10 --clients 10
```

> 💡 **AI Infra 视角**：用多进程客户端做压测，是为了模拟真实场景下多个用户同时请求的情况。观察指标时要注意区分吞吐量（throughput，单位时间完成的请求数）和平均延迟（latency），两者通常此消彼长——并发升高时吞吐上升、延迟也上升。生产容量规划时要同时盯住 p99 延迟和吞吐，而不是只看平均值。

## 已知问题与限制

1. 与本示例所基于的 [demo][demo_reference] 不同，diffusion 后端目前还不支持使用可选的 refiner（精修）模型。另见
   [demo_txt2img_xl.py][demo_code]


[demo_code]: https://github.com/NVIDIA/TensorRT/blob/release/10.4/demo/Diffusion/demo_txt2img_xl.py


[demo_reference]: https://github.com/NVIDIA/TensorRT/tree/release/10.4/demo/Diffusion#generate-an-image-with-stable-diffusion-xl-guided-by-a-single-text-prompt
