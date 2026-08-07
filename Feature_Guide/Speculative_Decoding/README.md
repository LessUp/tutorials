<!--
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 投机解码（Speculative Decoding）

- [关于投机解码](#关于投机解码)
- [性能提升](#性能提升)
- [Triton Inference Server 中的投机解码](#triton-inference-server-中的投机解码)

## 关于投机解码

投机解码（Speculative Decoding，又称 Speculative Sampling）是一组旨在让模型在一次前向传播（forward pass）迭代中生成多个 token 的技术。**在 GPU 因 batch size 较小而未充分利用的情况下**，它可以降低平均单 token 延迟。

投机解码的核心思路是：先用一种比反复执行目标大语言模型（LLM）高效得多的方法，预测出一段未来的 token 序列（称为草稿 token，draft tokens）；随后让目标 LLM 在单次前向传播中一次性验证这些草稿 token。其背后基于两个假设：

1. 并发处理多个草稿 token 的速度与处理单个 token 相当；
2. 在整个生成过程中，多个草稿 token 能够成功通过验证。

> 💡 **AI Infra 视角**：投机解码解决的是 LLM 推理中典型的「memory-bound」瓶颈——尤其在 decode 阶段、batch 较小时，GPU 计算资源大量空闲，瓶颈在于逐 token 串行生成。草案模型（draft model）用极小代价预测多个 token，目标模型（target model）一次前向传播批量验证，等于用「多算一点、少等一点」的思路换取更低的 TPOT（单 token 输出延迟）。这也是为什么在低并发、单请求场景下效果最明显。

如果第一个假设成立，投机解码的延迟就不会比标准解码差；如果第二个假设成立，每次前向传播平均推进的 token 数就能超过 1。两个假设同时满足时，投机解码就能带来可观的延迟下降。

## 性能提升

需要注意的是，投机解码技术的有效性高度依赖于具体任务。例如，在代码补全场景中预测后续 token，可能比为一篇文章生成摘要更简单。[Spec-Bench](https://sites.google.com/view/spec-bench) 展示了不同投机解码方法在不同任务上的表现差异。

> 💡 **AI Infra 视角**：评估投机解码收益的关键指标是「接受率」（acceptance rate，即草稿 token 被目标模型验证通过的比例）。接受率越高，单次前向传播推进的 token 越多，加速越明显；而接受率与任务本身的「可预测性」强相关。生产环境做加速收益评估时，务必用线上真实请求分布压测，而不是只看理想化的 benchmark——同时要留意投机解码多出的草稿计算会占用 GPU 算力，在并发高、batch 大时可能反而得不偿失。

## Triton Inference Server 中的投机解码

Triton Inference Server 支持在不同类型的 Triton backend 上使用投机解码。关于 Triton backend 是什么，请参见[这里](https://github.com/triton-inference-server/backend)。

- 点击[这里](TRT-LLM/README.md)了解 Triton Inference Server 如何配合 [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend) 支持投机解码。
- 点击[这里](vLLM/README.md)了解 Triton Inference Server 如何配合 [vLLM Backend](https://github.com/triton-inference-server/vllm_backend) 支持投机解码。
