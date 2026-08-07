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


# 使用 Model Analyzer 定制部署（Customizing deployment with Model Analyzer）

| 跳转到 | [第 2 部分：提升资源利用率](../Part_2-improving_resource_utilization/) | [第 4 部分：模型加速](../Part_4-inference_acceleration/) |
| ------------ | --------------- | --------------- |


每个推理部署都有一组独特的挑战。这些挑战可能来自关于延迟的 SLA（服务等级协议）、有限的硬件资源、单个模型的特殊需求、请求的性质和数量，或者其他完全不同的原因。此外，Triton Inference Server 有很多特性可以用来在内存占用和性能之间做权衡。

面对数量众多的特性和需求，为每个部署找到最优配置就变成了一项"扫一遍所有可能的配置来测量性能"的工作。本部分将讨论：
* 性能讨论
* 使用 Model Analyzer 找到最优配置

## 性能讨论（Performance Discussion）

为推理服务架构测量性能是一个相当复杂的问题。复杂性源于"运行推理"只是整个链条中的一环。为了理解这一点，我们走一遍：启用动态批处理、使用多个模型实例时，Triton Inference Server 会如何处理一条查询。

![Triton Architecture](./img/arch.jpg)

查询从客户端发出后，Triton 的处理器（handler）把它排入所请求模型的队列。一旦某个模型实例空闲，就会利用该查询、其他正在到达的查询或已在队列中的查询，形成一个与首选批大小（preferred batch size）对应的动态批次。这个批次随后被转换成框架所需的格式，发送给框架运行时（PyTorch、TensorFlow、TensorRT 等）。推理完成后，结果返回给客户端。

在这个过程中，有三个主要的延迟来源：
* 网络延迟（Network Latency）
* 推理计算时间（Inference Compute time）
* 模型队列中的等待时间（Latency caused due to wait time in the model's queue）

最小化**网络延迟**是一个因情况而异的过程。比如考虑计算机视觉模型：这些模型使用图像帧、视频和点云等 3D 数据，数据量可能很大，因此需要更高的带宽来传输。大多数图像以 float32 格式保存，可以转换为 float16 格式。这可能影响图像的动态范围，从而可能影响模型性能（取决于训练时采用的预处理步骤），但绝对可以减少延迟，因为需要传输的数据更少了。

加速模型以压缩实际**计算时间**通常通过大量技术实现，比如：融合层来优化网络图、降低模型精度、融合内核等等！这个话题在本系列第 4 部分有更深入的讨论。

**队列**中的延迟主要可以通过增加模型实例来解决。根据当前实例数量的 GPU 利用率情况，这可能不一定会带来额外的资源需求。这是每个部署环境都需要专门解决的核心资源利用问题。为了简化这一流程，Triton Inference Server 自带 Model Analyzer。

> 💡 **AI Infra 视角**：把推理延迟拆成"网络 + 计算 + 排队"三段是性能排查的基本功。真实线上调优的常见流程是：先用 perf_analyzer 看端到端延迟构成（server 侧日志会打印 queue/compute 细分），如果 queue 占比高说明实例不够或批策略待调，如果 compute 占比高则要去做模型加速（第 4 部分）。不要一上来就加 GPU，先分清瓶颈在哪一段。

Model Analyzer 是一个 CLI 工具，通过扫描各种配置设置并生成汇总性能的报告，帮助你更好地了解 Triton Inference Server 模型的计算和内存需求。

使用 Model Analyzer，用户可以：
* 运行可定制的配置扫描，为预期工作负载和硬件找出最佳配置。
* 通过详细的报告、指标和图表，汇总关于延迟、吞吐、GPU 资源利用率、功耗等方面的发现。这些报告有助于比较不同配置之间的性能表现。
* 让模型部署满足用户的服务质量（QoS）要求，比如特定的 p99 延迟上限、GPU 内存利用率以及最低吞吐量！

## 使用 Model Analyzer

### 前置条件

请参考本系列第 2 部分获取模型。关于安装 Model Analyzer 的更多信息，请参考 Model Analyzer 的[安装指南](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/install.md#recommended-installation-method)。为方便跟做，请使用以下命令安装 model analyzer：

```
sudo apt-get update && sudo apt-get install python3-pip
sudo apt-get update && sudo apt-get install wkhtmltopdf
pip3 install triton-model-analyzer
```

### 用法细节

在用示例深入细节之前，先讨论一下整体功能与能力，以便理解如何最好地使用 Model Analyzer 工具。我们从用户最关心的部分开始讨论：为扫描设置 `objectives`（目标）和 `constraints`（约束）。

- **objectives（目标）**：用户可以根据自己的部署目标——吞吐、延迟，或针对特定资源约束——对结果排序。[了解更多](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#objective)。

    Model Analyzer 有两种模式：在线（Online）和离线（Offline）。在线模式下，用户可以为部署指定延迟预算以满足需求；离线模式下可以类似地指定最低吞吐量。[了解更多](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/cli.md#model-analyzer-modes)

- **constraints（约束）**：用户也可以把扫描的选择限制在吞吐、延迟或 GPU 内存利用率的特定要求上。[了解更多](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#constraint)

讨论完更宽泛的选择后，我们来谈使用 Model Analyzer 需要掌握的两个关键子命令：`profile` 和 `report`。这些命令的大多数设置可以通过 flag 指定，但有些需要构建配置文件。完整设置列表请参考文档的[这一节](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md)。

- **profile**：`profile` 用于运行基准测试扫描。在这里用户指定扫描空间细节，比如每个 GPU 的实例数、模型最大批大小的范围、最大 CPU 利用率、发送查询的批大小、发送给 Triton 的并发查询数量以及[更多](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#config-options-for-profile)。`profile` 运行这些扫描，记录每个配置的性能，并把运行结果保存为检查点。可以把这一步理解成简单地运行大量实验并记录数据点供分析。这一步大约需要 60-90 分钟。用户可以使用 `--run-config-search-mode quick` flag 进行更快速、配置更少的扫描。更多信息，或想要更快、更小的扫描，请参考[文档](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#config-options-for-profile)。

- **report**：`report` 子命令生成最佳配置的详细报告以及一份汇总。这些报告包含：
  - 一张图表，展示随发送给服务器的并发请求数增加时吞吐和延迟的变化（详细报告）
  - 一张 GPU 内存 vs 延迟、GPU 利用率 vs 延迟的图表（详细报告）
  - 一张表格，列出 p99 延迟、延迟的各组成部分、吞吐、GPU 利用率和 GPU 内存利用率，覆盖到 profiling 步骤中选择的最大并发请求数（默认为 1024）（详细报告）
  - 一张吞吐 vs 延迟图、一张 GPU 内存 vs 延迟图，以及一张比较最佳配置与用户所选默认配置高层细节的表格（汇总报告）

看完下一节的示例后，这些选择会更具体。

### 示例

考虑以 `10 ms` 的延迟预算部署文本识别模型。第一步是给模型做 profile。这个命令会启动一轮扫描并记录性能。

`model-analyzer profile --model-repository /path/to/model/repository --profile-models <name of the model> --triton-launch-mode=<launch mode: local/docker etc> --output-model-repository-path /path/to/output -f <path to config file> --run-config-search-mode quick`

注意：配置文件包含查询图片的形状。关于 launch mode flag 的更多信息，请参考启动模式[文档](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/launch_modes.md)。

```
model-analyzer profile --model-repository /workspace/model_repository --profile-models text_recognition --triton-launch-mode=local --output-model-repository-path /workspace/output/ -f perf.yaml --override-output-model-repository --latency-budget 10 --run-config-search-mode quick
```

扫描完成后，用户可以使用 `report` 汇总最佳配置。

```
model-analyzer report --report-model-configs text_recognition_config_4,text_recognition_config_5,text_recognition_config_6 --export-path /workspace --config-file perf.yaml
```

生成的报告有两种类型：
* 汇总报告（Summaries）
* 详细报告（Detailed Reports）

**汇总报告**包含所有最佳配置的总体结果。它包含所用硬件的信息、吞吐 vs 延迟曲线、GPU 内存 vs 延迟曲线，以及一张包含性能数字和其他关键信息的表格。默认情况下，扫描空间被限制在一组常用特性上，如动态批处理和多个模型实例，但用户可以用[模型配置参数](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#model-config-parameters)把扫描空间扩展到 Triton 配置文件中可以指定的任何特性。

![summary](./img/report_1.PNG)

**详细报告**拆解每个配置的性能。它们包含更详细的性能图表，描述不同负载下的性能数字。

![summary](./img/report_2.PNG)

示例报告可以在 `reports` 文件夹中找到。

> 💡 **AI Infra 视角**：Model Analyzer 本质上做的是"自动化配置搜索"：把 instance 数、max_batch_size、dynamic batching 等参数组合成网格，自动拉起 Triton 跑 perf_analyzer 并收集指标。它解决的是生产里真实存在的痛点——配置空间太大、手工调参无法复现、不同模型的"最优配置"各不相同。做推理平台时，把它接入上线流程作为标准化的"性能体检"，比靠经验拍脑袋调参可靠得多。

# 接下来是什么？

本教程我们讲了 Model Analyzer 的用法，它是一个根据资源利用率选择最佳部署配置的工具。这是 6 部分教程系列的第 3 部分，该系列讨论的是将深度学习模型部署到生产环境所面临的挑战。第 4 部分讲的是 `推理加速（Inference Acceleration）`，会介绍框架层面的优化来加速你的模型！
