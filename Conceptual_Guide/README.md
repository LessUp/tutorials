<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# 概念指南（Conceptual Guides）

| 相关页面 | [Server 文档](https://github.com/triton-inference-server/server/tree/main/docs#triton-inference-server-documentation) |
| ------------ | --------------- |

概念指南（Conceptual Guide）系列是专为 Triton Inference Server 入门而设计的。本系列将涵盖以下内容：
* [第 1 部分：模型部署（Model Deployment）](Part_1-model_deployment/)：讲解如何部署和管理多个模型。
* [第 2 部分：提升资源利用率（Improving Resource Utilization）](Part_2-improving_resource_utilization/)：介绍在部署模型时最大化 GPU 利用率的两个常用特性/技术。
* [第 3 部分：优化 Triton 配置（Optimizing Triton Configuration）](Part_3-optimizing_triton_configuration/)：每个部署都有其用例特有的需求。本指南带用户走一遍如何定制部署配置以满足 SLA（服务等级协议）要求。
* [第 4 部分：模型加速（Accelerating Models）](Part_4-inference_acceleration/)：获得更高吞吐的另一条路径是加速底层模型。本指南介绍可用于加速模型的 SDK 和工具。
* [第 5 部分：构建模型集成（Building Model Ensembles）](./Part_5-Model_Ensembles/)：模型很少被单独使用。本指南讲解"如何构建一条深度学习推理流水线"。
* [第 6 部分：使用 BLS API 构建复杂流水线](Part_6-building_complex_pipelines/)：很多时候流水线需要控制流（conditional logic）。了解如何与部署在不同后端上的模型一起构建复杂流水线。
* [第 7 部分：迭代调度教程（Iterative Scheduling Tutorial）](./Part_7-iterative_scheduling)：演示如何配合 HuggingFace Transformers，用 Triton 迭代调度器（Iterative Scheduler）部署 GPT2 模型。
* [第 8 部分：语义缓存（Semantic Caching）](./Part_8-semantic_caching/)：展示为基于 LLM 的工作流加入语义缓存带来的收益。

> 💡 **AI Infra 视角**：这个系列按"部署 → 提性能 → 调配置 → 加速 → 编排"的路径组织，基本对应真实推理平台的搭建顺序。作为 AI Infra 工程师，建议按顺序通读前 4 篇（涉及 model repository、batching、实例组、Model Analyzer 等核心机制），第 5-8 篇则对应流水线编排、动态调度等进阶场景，可按需选读。
