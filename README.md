<!--
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Triton 教程（Triton Tutorials）

对于习惯"张量进、张量出"这种深度学习推理方式的用户来说，上手 Triton 时往往会产生很多疑问。本仓库的目标是帮助用户熟悉 Triton 的各项特性，并提供指南与示例来简化迁移过程。如需逐特性的讲解，请参阅 [Triton Inference Server 官方文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)。

> 💡 **AI Infra 视角**：Triton 在整个 AI 推理链路中扮演"推理网关"（serving layer）的角色——上游接各类框架导出的模型（PyTorch、ONNX、TensorRT 等），下游对客户端暴露统一的 HTTP/gRPC 接口。做 AI Infra 时理解这一点很关键：模型训练和推理服务是两套体系，Triton 解决的是"模型如何稳定、高效、规模化地对外服务"这一层的问题。

#### 入门清单（Getting Started Checklist）
| [概览视频（Overview Video）](https://www.youtube.com/watch?v=NQDtfSi5QF4) | [概念指南：模型部署（Conceptual Guide: Deploying Models）](Conceptual_Guide/Part_1-model_deployment/README.md) |
| ------------ | --------------- |

## 快速部署（Quick Deploy）

这些示例的重点是演示如何部署用各种框架训练出的模型。它们是快速演示，前提是用户已对 Triton 有一定了解。

### 部署一个……
| [PyTorch 模型](./Quick_Deploy/PyTorch/README.md) | [TensorFlow 模型](./Quick_Deploy/TensorFlow/README.md) | [ONNX 模型](./Quick_Deploy/ONNX/README.md) | [TensorRT 加速模型](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton) | [vLLM 模型](./Quick_Deploy/vLLM/README.md) | [OpenVINO 模型](./Quick_Deploy/OpenVINO/README.md)
| --------------- | ------------ | --------------- | --------------- | --------------- | --------------- |

## LLM 教程（LLM Tutorials）
下表列出的是本教程中支持的一些热门模型：
| 示例模型（Example Models）   | 教程链接（Tutorial Link） |
| :-------------: | :------------------------------: |
| [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) |[TensorRT-LLM 教程](Popular_Models_Guide/Llama2/trtllm_guide.md) |
| [Persimmon-8B](https://www.adept.ai/blog/persimmon-8b) | [HuggingFace Transformers 教程](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/HuggingFaceTransformers)  |
| [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) |[HuggingFace Transformers 教程](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/HuggingFaceTransformers)   |
| [LLaVA-v1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | [TensorRT-LLM 教程](Popular_Models_Guide/Llava1.5/llava_trtllm_guide.md) |

## 生成式推荐（Generative Recommenders）
| 示例模型（Example Models） | 教程链接（Tutorial Link） |
| :-------------: | :------------------------------: |
| [HSTU](https://github.com/NVIDIA/recsys-examples/tree/main/examples/hstu) | [Triton 上的 HSTU（Torch AOTI）](Popular_Models_Guide/HSTU/README.md) |

**注意（Note）：**
这份清单并不穷尽 Triton 支持的全部能力，只是本教程仓库中收录的内容。

## 构建 Triton（Building Triton）

针对特定平台从源码构建 Triton Inference Server 本身的指南。

| [为 RHEL / manylinux 构建（示例）](./Build_Guide/RHEL_Manylinux/README.md) |
| --------------- |

## 本仓库包含什么？

本仓库包含以下资源：
* [概念指南（Conceptual Guide）](./Conceptual_Guide/)：本指南专注于帮用户建立对推理基础设施建设中常见挑战的概念性理解，以及如何用 Triton Inference Server 最好地应对这些挑战。
* [快速部署（Quick Deploy）](#quick-deploy)：这是一组指南，讲解如何将首选框架的模型部署到 Triton Inference Server 上。这些指南假定读者已对 Triton Inference Server 有基本了解。建议先完整过一遍入门材料以获得更全面的认识。
* [HuggingFace 指南](./HuggingFace/)：本指南重点带领用户走一遍用 Triton Inference Server 部署 HuggingFace 模型的各种方式。
* [特性指南（Feature Guides）](./Feature_Guide/)：此目录用于存放 Triton 特定功能的示例。
* [迁移指南（Migration Guide）](./Migration_Guide/migration_guide.md)：正在从现有方案迁移到 Triton Inference Server？先了解什么样的整体架构最适合你的场景。
* [构建指南（Build Guide）](./Build_Guide/)：针对特定平台（如 RHEL / manylinux）从源码构建 Triton Inference Server 的社区示例。这些不是官方支持的构建路径。

## 导航 Triton Inference Server 资源

Triton Inference Server GitHub 组织下包含多个仓库，承载着 Triton Inference Server 的不同功能。以下并不是所有仓库的完整描述，只是一个帮助建立直观理解的简易指南。

* [Server](https://github.com/triton-inference-server/server) 是 Triton Inference Server 的主仓库。
* [Client](https://github.com/triton-inference-server/client) 包含创建 Triton 客户端所需的库和示例。
* [Backend](https://github.com/triton-inference-server/backend) 包含构建新 Triton Backend 的核心脚本和工具。任何名称中带 "backend" 的仓库，要么是某个框架的后端，要么是教你如何创建后端的示例。
* [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) 和 [Model Navigator](https://github.com/triton-inference-server/model_navigator) 等工具分别用于性能测量和简化模型加速流程。

> 💡 **AI Infra 视角**：这些配套工具是生产环境里真正会天天用的：Model Analyzer 帮你做配置空间搜索（选最优的 batch 大小、instance 数量），Model Navigator 帮你自动做模型转换与格式探索（PyTorch → ONNX → TensorRT）。在团队里做推理平台时，把这些工具集成进 CI 流程，能让"上线前必做的压测和调优"从手工操作变成自动化环节。

## 提交请求（Adding Requests）

如果你希望新增某个示例，请开一个 issue 并写明需求细节。想贡献代码？开一个 pull request 并 @ 管理员即可。
