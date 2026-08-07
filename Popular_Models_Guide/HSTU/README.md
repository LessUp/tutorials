<!--
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 在 Triton 上部署 HSTU 生成式推荐模型

[Hierarchical Sequential Transduction Units (HSTU)](https://arxiv.org/abs/2402.17152)
驱动着**生成式推荐器（Generative Recommenders，GRs）**：把推荐任务重构为对高基数、非平稳事件流（event streams）的生成式建模。HSTU
同时支持检索（retrieval）和排序（ranking）两类任务。

> 💡 **AI Infra 视角**：传统推荐系统通常走"召回（两阶段/多路）→ 精排 → 重排"的级联管线，每个环节是独立训练的模型；生成式推荐则把整个推荐问题统一建模成一个生成任务——用用户的点击/浏览行为序列直接预测下一个交互目标，架构更简单、信息利用更充分。HSTU 是这一方向的代表性架构，其 Transformer 式结构天然适合 GPU 推理，这也是它能端到端地跑在 Triton 上的原因。

Triton Inference Server 可以通过
[PyTorch backend](https://github.com/triton-inference-server/pytorch_backend)
的 ahead-of-time（AOT）Inductor 包（`platform: "torch_aoti"`）来服务 HSTU 模型。训练、
导出、KV-cache 运行时以及端到端示例都位于 NVIDIA 的
[recsys-examples](https://github.com/NVIDIA/recsys-examples) 仓库中，而不是本 tutorials 仓库。

> 💡 **AI Infra 视角**：AOTInductor 是 PyTorch 2 的编译能力之一：在部署前（ahead-of-time）就把模型编译成不依赖完整 PyTorch 运行时的原生算子包，服务时省去 Python 解释和 eager 模式的调度开销。对推荐这类对延迟敏感且请求量巨大的在线服务来说，这种"预编译换取低延迟"的思路与 TensorRT 引擎异曲同工，但生态上完全围绕 PyTorch，模型迭代成本更低。

## 下一步去哪里

| 资源 | 说明 |
| -------- | ----------- |
| [HSTU 概览](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/README.md) | 架构、训练与推理的入口 |
| [HSTU 推理](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/inference/README.md) | 推理特性、KV-cache、AOTInductor 导出及 KuaiRand 示例 |
| [Triton 上的 PyTorch AOTI](https://github.com/triton-inference-server/pytorch_backend#aot-inductor-support-beta) | `torch_aoti` 模型仓库布局与配置 |

> [!NOTE]
> HSTU 模型的构建、导出与验证请使用 recsys-examples 中的指南。本页面只是为 Triton 用户指引该工作流，以及 Torch AOTI 的服务路径。
