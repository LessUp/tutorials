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

# Triton Inference Server Ray Serve 部署

借助 Triton Inference Server 进程内 Python API，你可以把基于 Triton 的模型集成到任意 Python 框架中，包括 FastAPI 和 Ray Serve。

本目录包含一个基于 FastAPI 的 Triton Inference Server Ray Serve 部署示例。

> 💡 **AI Infra 视角**：这里 Triton 与 Ray Serve 的分工很有代表性：Ray Serve 充当编排层（orchestration layer），负责 HTTP 入口、请求路由和弹性扩缩容；Triton 则是其中的推理引擎，管理模型加载与 GPU 执行。类似的组合还有 Triton + Kubernetes + 服务网格。理解「编排层管弹性、推理引擎管算力」这条职责边界，是设计生产级推理系统的基础。

| [安装](#安装) | [运行部署](#运行-ray-serve-部署) | [发送请求](#向部署发送请求) |


## 安装

Stable Diffusion 流水线基于 [Popular_Models_Guide/StableDiffusion](../../../Popular_Models_Guide/StableDiffusion) 教程。

### 克隆仓库
```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Triton_Inference_Server_Python_API
```

### 构建 Tritonserver 镜像与 Stable Diffusion 模型

请注意，以下命令可能需要数分钟甚至更久，具体取决于你的硬件配置和网络连接。

```bash
./build.sh --framework diffusion --build-models
```

## 运行 Ray Serve 部署

### 启动容器

以下命令会启动一个容器，并把当前目录挂载为 `workspace`。

```bash
./run.sh --framework diffusion
cd examples/rayserve
```

### 启动本地 Ray 集群

以下命令会启动一个本地 Ray 集群，同时启动 Prometheus 和 Grafana 实例，并启用默认的 Ray 与 Ray Serve 指标和仪表盘。

```
./start_ray.sh
```

### 运行部署

> 💡 **AI Infra 视角**：`serve run tritonserver_deployment:deployment` 中的 `deployment()` 是这个文件的入口，返回 `TritonDeployment.bind()`——一个尚未实例化的部署对象。Ray Serve 的弹性伸缩（autoscaling）配置同样在 `@serve.deployment` 装饰器中声明：`min_replicas`/`max_replicas` 限定副本范围，`target_ongoing_requests` 决定按在途请求数扩缩容，属于典型的负载驱动型（load-based）弹性策略。

```bash
serve run tritonserver_deployment:deployment
```

## 向部署发送请求

该部署包含两个端点：

### `/identity`

identity 端点接收一个字符串并原样返回。

#### 示例请求
```
curl --request GET "http://127.0.0.1:8000/identity?string_input=hello_world!"
```

#### 示例输出
```bash
"hello_world!"
```

### `/generate`
generate 端点接收一个提示词（prompt），用 stable diffusion 根据提示词生成图片，并把图片保存到文件。

#### 示例请求
```
curl --request GET "http://127.0.0.1:8000/generate?prompt=car,model-t,realistic,4k&filename=/workspace/examples/rayserve/car_sample.jpg"
```

#### 示例输出

![car_sample](../../docs/car_sample.jpg)


## 查看 Ray 与 Ray Serve 仪表盘

Ray 与 Ray Serve 仪表盘托管在默认端口上，可用于可视化各项指标：

> 💡 **AI Infra 视角**：Ray 自带的仪表盘（默认端口 8265）直接暴露推理延迟、副本数、GPU 利用率等指标，并预置 Prometheus/Grafana 支持。对 AI Infra 从业者来说，推理服务上线后的可观测性（observability）与推理本身同等重要——吞吐、延迟分位数、排队长度这些指标，决定了你能否在流量增长时及时扩容，以及在 SLA 违约之前发现问题。

```
<IP_ADDRESS>:8265
```

## 停止 Ray Serve 集群

以下命令会停止本地 Ray 集群，同时停止 Prometheus 和 Grafana 实例。


```bash
./stop_ray.sh
```
