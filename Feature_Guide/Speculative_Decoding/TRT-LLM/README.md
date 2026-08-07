<!--
# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 使用 TensorRT-LLM 的投机解码（Speculative Decoding with TensorRT-LLM）

- [关于投机解码](#关于投机解码)
- [EAGLE-3](#eagle-3)
- [MEDUSA](#medusa)
- [基于草稿模型的投机解码](#基于草稿模型的投机解码)

## 关于投机解码

本教程演示如何在单节点单 GPU 上，通过 [TensorRT-LLM LLM API / PyTorch backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llmapi.md) 在 Triton Inference Server 中构建并部署投机解码模型。其他受支持的 backend 请参见[投机解码](../README.md)主页。

> **注意：** 本教程使用现代的 **LLM API / PyTorch backend**，它直接读取 HuggingFace 模型权重（checkpoint），无需构建 TensorRT engine。如果你需要旧版的基于 TRT engine 的方案，参见 [engine backend archive](https://github.com/triton-inference-server/tensorrtllm_backend#tensorrt-engine-backend)。

根据 [Spec-Bench](https://sites.google.com/view/spec-bench) 的结果，EAGLE 是目前加速 LLM 推理表现最好的方法。
本教程将重点介绍 [EAGLE-3](#eagle-3)，演示如何让它在 Triton Inference Server 上工作，同时也会介绍[基于草稿模型的投机解码](#基于草稿模型的投机解码)，供想尝试其他方案的读者参考。你可以根据自己的需求选择最合适的方案。

> 💡 **AI Infra 视角**：TensorRT-LLM 同时提供「LLM API / PyTorch backend」与「TRT engine backend」两条部署路径。前者直接跑 HuggingFace checkpoint，省去编译 engine 的流程，迭代模型版本只需替换权重，非常适合快速验证和频繁换模型；后者需要把模型编译成 TRT engine，启动慢但推理性能更极致。生产环境若追求极致吞吐且模型固定，仍可考虑 engine 方案。

## EAGLE-3

EAGLE-3（[论文](https://arxiv.org/pdf/2503.01840) | [GitHub](https://github.com/SafeAILab/EAGLE) | [博客](https://sites.google.com/view/eagle-llm)）是 EAGLE 投机解码技术的最新一代：它基于上下文特征预测未来 token，从而加速大语言模型（LLM）推理。EAGLE-3 使用一个轻量级的草稿头（draft head）预测下一个特征向量，再通过 LLM 冻结的分类头生成 token，相比普通解码可获得 2-3 倍加速，同时保持输出质量。与 EAGLE（v1/v2）相比，EAGLE-3 通过训练时测试增强进一步提升了接受率。

### 下载目标模型和草稿模型（可选）

本示例使用 [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 作为目标模型，[yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) 作为草稿模型。两个模型都可以在服务器启动时通过挂载 HuggingFace 缓存目录来自动下载；你也可以预先手动下载：

```bash
# Authenticate first if needed (Llama-3.1 requires accepting the license on HuggingFace)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

更多兼容 EAGLE-3 的草稿模型权重可以在 NVIDIA 的 [Speculative Decoding Modules](https://huggingface.co/collections/nvidia/speculative-decoding-modules) 合集中找到。

### 启动 Triton TensorRT-LLM 容器

启动带 TensorRT-LLM backend 的 Triton 容器。挂载你的 HuggingFace 缓存目录，让模型可以自动下载。把 `<xx.yy>` 替换为你想要使用的 Triton 版本（必须 >= 25.01）。推荐使用最新的 Triton Server 容器，可在[这里](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)找到。

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

### 准备模型仓库

在容器内复制 LLM API 模型模板：

```bash
cp -R /app/all_models/llmapi/ /opt/tritonserver/llmapi_repo/
```

编辑 `/opt/tritonserver/llmapi_repo/tensorrt_llm/1/model.yaml` 来配置 EAGLE-3：

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
  speculative_model: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

triton_config:
  max_batch_size: 0
  decoupled: False
```

*注意：如果你已经预先下载了模型，`model` 和 `speculative_model` 也可以填本地文件系统路径。*

### 用 Triton 部署

使用 [launch_triton_server.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/scripts/launch_triton_server.py) 脚本启动 Triton Server：

```bash
python3 /app/scripts/launch_triton_server.py --model_repo=/opt/tritonserver/llmapi_repo/
```

> 服务器就绪后你应该会看到类似下面的日志：
> ```
> I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
> I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
> I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
> ```

在容器内停止 Triton Server，运行：
```bash
pkill tritonserver
```
*注意：如果因为各种原因导致 Triton Server 启动失败，别忘了运行上面的命令停止它，否则可能引发 OOM 或 MPI 问题。*

### 发送推理请求

你可以用 [generate 端点](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html) 测试运行结果：

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
  -d '{"text_input": "What is ML?", "sampling_param_max_tokens": 50}'
```

> 你应该会收到类似下面的响应：
> ```json
> {"model_name":"tensorrt_llm","model_version":"1","text_output":"What is ML?\nML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."}
> ```

（可选）在请求中加上 `"sampling_param_return_perf_metrics": true`，即可返回投机解码的性能指标：

```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate \
  -d '{"text_input": "What is ML?", "sampling_param_max_tokens": 50, "sampling_param_return_perf_metrics": true}' | jq
```

响应中会多出 `acceptance_rate`（接受率）、`total_accepted_draft_tokens`（被接受的草稿 token 总数）和 `total_draft_tokens`（草稿 token 总数）等字段，这些字段对评估投机解码的效果很有用。

> 💡 **AI Infra 视角**：`acceptance_rate` 是线上诊断投机解码效果的第一抓手。若它明显低于理想值（比如 EAGLE 系通常可达 0.7-0.9），先检查草稿模型与目标模型是否匹配、`max_draft_len` 是否过大；接受率过低时投机解码的额外计算纯属浪费，甚至可能让整体延迟不降反升。把这些指标接入监控告警，比事后对比基准更有价值。

### 用 Gen-AI Perf 评估性能

Gen-AI Perf 是一个命令行工具，用于测量推理服务器上生成式 AI 模型的吞吐和延迟。更多信息见[这里](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)。我们将用 Gen-AI Perf 评估 EAGLE-3 相对基座模型的性能提升。

*注意：下面的实验在单节点单 GPU（RTX 5880，48GB 显存）上完成。以下数字仅供参考，实际数值会因硬件和环境不同而有所差异。*

1. 准备数据集

   我们使用 EAGLE 原论文所用的 HumanEval 数据集进行评估。该数据集已转换为 EAGLE 需要的格式，可在[这里](https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/humaneval/question.jsonl)获取。为了兼容 Gen-AI Perf，还需要再做一次转换。你也可以使用 HumanEval 之外的其他数据集，只要它能转换成 Gen-AI Perf 要求的格式即可。注意 MT-bench 目前不能使用，因为 Gen-AI Perf 还不支持多轮（multiturn）数据集作为输入。按以下步骤下载并转换数据集。

   ```bash
   wget https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/data/humaneval/question.jsonl

   # dataset-converter.py file can be found in the parent folder of this README.
   python3 dataset-converter.py --input_file question.jsonl --output_file converted_humaneval.jsonl
   ```

2. 安装 GenAI-Perf（Ubuntu 24.04，Python 3.10+）

   ```bash
   pip install genai-perf
   ```
   *注意：必须已经安装 CUDA 12。*

3. 运行 Gen-AI Perf

   在 SDK 容器中执行以下命令：
   ```bash
   genai-perf \
     profile \
     -m tensorrt_llm \
     --service-kind triton \
     --backend tensorrtllm \
     --input-file /path/to/converted/dataset/converted_humaneval.jsonl \
     --tokenizer meta-llama/Llama-3.1-8B-Instruct \
     --profile-export-file my_profile_export.json \
     --url localhost:8001 \
     --concurrency 1
   ```
   *注意：在对比投机解码与基座模型的加速效果时，请使用 `--concurrency 1`。这个设置很关键，因为投机解码的本质是用额外的计算换取更低的 token 生成延迟。限制并发可以避免多个请求占满硬件资源，从而更准确地评估该技术在延迟上的收益。这样才能确保 benchmark 反映投机解码在真实低并发场景下的性能增益。*

4. 在基座模型上运行 Gen-AI Perf

   为了对比 EAGLE-3 与基座模型（即未启用投机解码的普通 LLM）的性能，用去掉 `speculative_config` 配置块的 `model.yaml` 重启 Triton Server：

   ```yaml
   model: meta-llama/Llama-3.1-8B-Instruct
   backend: pytorch

   tensor_parallel_size: 1
   pipeline_parallel_size: 1

   triton_config:
     max_batch_size: 0
     decoupled: False
   ```

   然后重新运行上面的 Gen-AI Perf 命令。

5. 对比性能

   从示例运行来看，在低并发下 EAGLE-3 的 token 吞吐通常比基座模型提升 2 倍或更多。具体加速比因硬件、模型和数据集而异。

   如上所述，上面的数据来自单节点单 GPU（RTX 5880，48GB 显存）。实际数值会因硬件和环境不同而有所差异。

## MEDUSA

> **重要：** MEDUSA 在**现代 LLM API / PyTorch backend 中不受支持**，只能在旧版 TRT engine backend 下工作。
>
> 新部署建议改用 [EAGLE-3](#eagle-3)，它在 LLM API / PyTorch backend 上得到完整支持，且草稿准确率更高。
>
> 如果你确实需要 MEDUSA 配合 TRT engine backend，请参阅 [engine backend archive](https://github.com/triton-inference-server/tensorrtllm_backend#tensorrt-engine-backend) 中的旧版说明。

## 基于草稿模型的投机解码

基于草稿模型的投机解码（[论文](https://arxiv.org/pdf/2302.01318)）是另一种加速 LLM 推理的方法：使用一个更小、更快的 LLM 作为草稿模型，一次性预测多个后续 token。这种方法与 EAGLE-3 不同，在现代 LLM API / PyTorch backend 中受支持。与 EAGLE-3 相比的主要区别如下：

 - 草稿生成：使用一个独立的 LLM 作为草稿模型，一次性预测多个后续 token。这与 EAGLE-3 在目标模型内嵌轻量草稿头、从特征层面外推的方式形成对比。

 - 验证过程：草稿生成和验证采用链式（线性）结构，而 EAGLE-3 使用基于树的注意力机制。

 - 一致性：与 EAGLE-3 类似，在贪心（greedy）和非贪心设置下都能与目标 LLM 保持分布一致性。

 - 效率：虽然有效，但通常比 EAGLE-3 慢。

 - 实现：需要一个与目标模型共享相同 tokenizer 的独立草稿模型。草稿模型可以是任何 HuggingFace 兼容的 LLM。

要通过 LLM API 在 Triton 上使用基于草稿模型的投机解码，按照上面 [EAGLE-3](#eagle-3) 一节的方法完成容器启动和模型仓库准备，但把 `model.yaml` 配置成如下形式：

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
backend: pytorch

tensor_parallel_size: 1
pipeline_parallel_size: 1

speculative_config:
  decoding_type: Draft_Target
  max_draft_len: 3
  speculative_model: /path/to/draft_model  # Must share the same tokenizer as the target model

triton_config:
  max_batch_size: 0
  decoupled: False
```

*注意：草稿模型和目标模型必须使用同一个 tokenizer 训练。如果两者不兼容，接受率会极低，性能不升反降。*
