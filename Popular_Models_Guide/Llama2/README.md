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

# 在 Triton 中部署 Hugging Face Transformer 模型

用 Tritonserver 运行 Llama2 有多种方式：
1. 使用 [TensorRT-LLM Backend](trtllm_guide.md) 推理
2. 使用 [vLLM Backend](vllm_guide.md) 推理
3. 使用 [基于 Python 的后端，以 HuggingFace 模型方式推理](../../Quick_Deploy/HuggingFaceTransformers/README.md#deploying-hugging-face-transformer-models-in-triton)

> 💡 **AI Infra 视角**：同一个模型在 Triton 上可以走三条完全不同的推理路径，选择取决于你的目标：TensorRT-LLM 编译期深度优化，延迟最低、吞吐最高，但需要构建引擎、调参成本高；vLLM 开箱即用、持续批处理成熟，部署最快；纯 Python 后端直接加载 HuggingFace 权重，最适合快速原型和模型尚未定型时。生产实践上，通常先用方案 3 验证效果，稳定后迁移到方案 1 或 2 追求性能。

## 预构建说明

本教程假定 Llama2 模型、权重和分词器（tokenizer）已从 HuggingFace 的 Llama2 仓库[克隆](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)。
运行教程前，你需要获得 Llama2 仓库的访问权限，并能使用 huggingface cli。
该 cli 使用[用户访问令牌](https://huggingface.co/docs/hub/security-tokens)。令牌可以在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 获取。

> 💡 **AI Infra 视角**：很多开源模型（尤其是 Llama 这类需要申请授权的模型）下载都要访问令牌，这在生产部署里是一个容易踩坑的工程问题：CI 里不能硬编码令牌，常见的做法是用密钥管理系统（如 Vault、K8s Secret）注入 `HF_TOKEN` 环境变量。另一个生产经验是提前把权重下载好并固化版本、离线分发到目标机器，避免服务启动时才联网拉取——既防止网络抖动导致启动失败，也能精确锁定权重版本，保证线上与测试环境一致。
