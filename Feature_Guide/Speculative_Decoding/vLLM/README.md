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

# 使用 vLLM 的投机解码（Speculative Decoding with vLLM）

- [关于投机解码](#关于投机解码)
- [EAGLE](#eagle)
- [基于草稿模型的投机解码](#基于草稿模型的投机解码)

## 关于投机解码

本教程演示如何在单节点单 GPU 上，通过 [vLLM Backend](https://github.com/triton-inference-server/vllm_backend) 在 Triton Inference Server 中构建并部署投机解码模型。其他受支持的 backend 请参见[投机解码](../README.md)主页。

根据 [Spec-Bench](https://sites.google.com/view/spec-bench) 的结果，EAGLE 是目前加速 LLM 推理表现最好的方法。本教程将重点介绍 [EAGLE](#eagle)，演示如何让它在 Triton Inference Server 上工作，同时也会介绍[基于草稿模型的投机解码](#基于草稿模型的投机解码)，供想尝试其他方案的读者参考。想了解 vLLM 内部如何支持投机解码，可参见[这篇博客](https://blog.vllm.ai/2024/10/17/spec-decode.html)。完成本教程后，你就能在 Triton Inference Server 上轻松尝试 vLLM 提供的其他投机解码技术（参见[官方文档](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculative-decoding)）。

> 💡 **AI Infra 视角**：EAGLE 与普通草稿模型方案的本质区别在于「草稿从哪来」。传统方案用一个小 LLM 独立生成草稿 token，而 EAGLE 复用目标模型倒数第二层输出的特征向量，用轻量的自回归头（auto-regression head）预测「下一个特征」，再经目标模型冻结的分类头（classification head）映射回 token。因为特征空间比 token 空间信息量大得多，草稿预测更准（接受率更高），同时额外参数量极小，几乎不挤占显存。

## EAGLE

EAGLE（[论文](https://arxiv.org/pdf/2401.15077) | [GitHub](https://github.com/SafeAILab/EAGLE) | [博客](https://sites.google.com/view/eagle-llm)）是一种投机解码技术：它基于 LLM 倒数第二层提取的上下文特征来预测未来 token，从而加速大语言模型（LLM）推理。EAGLE 使用一个轻量级的自回归头（Auto-regression Head）预测下一个特征向量，再通过 LLM 冻结的分类头生成 token，相比普通解码可获得 2-3 倍加速，同时保持输出质量和分布一致性。

### 获取 EAGLE 模型及其基座模型

本示例使用 [EAGLE-LLaMA3-Instruct-8B](https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B) 模型。更多 EAGLE 模型见[这里](https://huggingface.co/yuhuili)。EAGLE 正常工作还需要基座模型 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。

下载两个模型的命令如下：
```bash
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git clone https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```
*注意：Llama3 系列模型需要在 Hugging Face 上申请访问权限并登录后才能下载和使用。*

### 转换 EAGLE 模型

根据 vLLM 官方[文档](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-using-eagle-based-draft-models)：
> ... 自 [PR 12304](https://github.com/vllm-project/vllm/pull/12304) 之后，EAGLE 模型应该可以直接被 vLLM 加载使用。如果你使用的 vllm 版本早于 [PR 12304](https://github.com/vllm-project/vllm/pull/12304)，请使用这个[脚本](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)转换投机模型，并指定 speculative_model="path/to/modified/eagle/model" ...

对 Triton 而言，如果你使用的 Triton Server 容器版本 <= 25.02，需要在同时包含 EAGLE 模型和基座模型的目录下运行上述[脚本](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)来转换 EAGLE 模型。Triton Server 容器版本 >= 25.03 使用的 vLLM 版本（>= 0.7.3）已包含 PR 12304。

### 创建模型仓库（Model Repository）

模型仓库是 Triton 读取模型及其元数据（配置、版本文件等）的方式。关于模型仓库的细节，参见[这里](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_1-model_deployment/README.html#setting-up-the-model-repository)。

我们在 [model_repository](model_repository) 中准备了 EAGLE 模型和基座模型的模板，请复制后按需修改其中的 model.json。例如，参照 vLLM [示例](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-with-a-draft-model)，我们将 eagle_model 的 `num_speculative_tokens` 设为 5。你也可以改成其他值，这会影响性能。

> 💡 **AI Infra 视角**：`num_speculative_tokens`（草稿长度）是投机解码最核心的调参旋钮之一。草稿越长，单次前向传播理论上能推进的 token 越多，但接受率会随草稿长度递减，且每轮验证都要重新计算 KV cache——草稿过长时，多出的验证开销反而会拖慢生成。实践中通常从 3-5 起步，用 Gen-AI Perf 对比不同取值下的 TPOT 和吞吐再定。

### 用 Triton 部署

下面启动带 vLLM backend 的 Triton docker 容器来部署模型。
注意我们把下载（以及可能转换过）的 EAGLE 和基座模型挂载到容器内的 `/hf-models`，把上一节准备的模型仓库挂载到 `/model_repository`。请将 <xx.yy> 替换为你想要使用的 Triton 版本。推荐使用最新的 Triton Server 容器，可在[这里](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)找到。

```bash
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v </path/to/model_repository>:/model_repository \
    -v </path/to/eagle/and/base/model>:/hf-models \
    nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 \
    tritonserver --model-repository /model_repository \
    --model-control-mode explicit --load-model eagle_model
```

### 发送推理请求

向 [generate 端点](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html)发送一个推理请求。

```bash
curl -X POST localhost:8000/v2/models/eagle_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}' | jq
```

> 你应该会收到类似下面的响应：
> ```
> {
>  "model_name": "eagle_model",
>  "model_version": "1",
>  "text_output": "What is Triton Inference Server?¶\n\nTriton Inference Server is an open-source, high-performance,"
> }
> ```

### 用 Gen-AI Perf 评估性能

Gen-AI Perf 是一个命令行工具，用于测量推理服务器上生成式 AI 模型的吞吐和延迟。更多信息见[这里](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)。我们将用 Gen-AI Perf 评估 EAGLE 模型相对基座模型的性能提升。

1. 准备数据集

我们使用 EAGLE 原论文所用的 HumanEval 数据集进行评估。该数据集已转换为 EAGLE 需要的格式，可在[这里](https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/humaneval/question.jsonl)获取。为了兼容 Gen-AI Perf，还需要再做一次转换。你也可以使用 HumanEval 之外的其他数据集，只要它能转换成 Gen-AI Perf 要求的格式即可。注意 MT-bench 目前不能使用，因为 Gen-AI Perf 还不支持多轮（multiturn）数据集作为输入。按以下步骤下载并转换数据集。
```bash
wget https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/data/humaneval/question.jsonl

# dataset-converter.py file can be found in the parent folder as this README.
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
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --input-file /path/to/converted/dataset/converted_humaneval.jsonl \
  --tokenizer /path/to/hf-models/Meta-Llama-3-8B-Instruct/ \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --concurrency 1
```
*注意：在对比投机解码与基座模型的加速效果时，请使用 `--concurrency 1`。这个设置很关键，因为投机解码的本质是用额外的计算换取更低的 token 生成延迟。限制并发可以避免多个请求占满硬件资源，从而更准确地评估该技术在延迟上的收益。这样才能确保 benchmark 反映投机解码在真实低并发场景下的性能增益。*

示例输出如下：
```
                                    NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│              Request Latency (ms) │ 7,510.69 │ 6,534.94 │ 8,433.33 │ 8,409.31 │ 8,193.07 │ 7,832.68 │
│   Output Sequence Length (tokens) │   325.00 │   324.00 │   326.00 │   325.97 │   325.70 │   325.25 │
│    Input Sequence Length (tokens) │   112.50 │    79.00 │   137.00 │   136.55 │   132.50 │   125.75 │
│ Output Token Throughput (per sec) │    43.27 │      N/A │      N/A │      N/A │      N/A │      N/A │
│      Request Throughput (per sec) │     0.13 │      N/A │      N/A │      N/A │      N/A │      N/A │
│             Request Count (count) │     4.00 │      N/A │      N/A │      N/A │      N/A │      N/A │
└───────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

*注意：上述示例输出来自单节点单 GPU（RTX 5880，48GB 显存）。以下数字仅供参考，实际数值会因硬件和环境不同而有所差异。*

4. 在基座模型上运行 Gen-AI Perf

为了对比 EAGLE 与基座模型（即未启用投机解码的普通 LLM）的性能，我们还需要在基座模型上运行 Gen-AI Perf。部署基座模型只需修改[用 Triton 部署](#用-triton-部署)一节，把 `--load-model` 参数从 `eagle_model` 换成 `base_model`：

```bash
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v </path/to/model_repository>:/model_repository \
    -v </path/to/eagle/and/base/model>:/hf-models \
    nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 \
    tritonserver --model-repository /model_repository \
    --model-control-mode explicit --load-model base_model
```

请谨慎使用 EAGLE，因为根据 vLLM [文档](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-using-eagle-based-draft-models)：

> 在 vLLM 中使用 EAGLE 类投机器时，实测加速比参考实现[这里](https://github.com/SafeAILab/EAGLE)报告的要低。该问题正在调查中，跟踪进展见 [vllm-project/vllm#9565](https://github.com/vllm-project/vllm/issues/9565)。

## 基于草稿模型的投机解码

基于草稿模型的投机解码（[论文](https://arxiv.org/pdf/2302.01318)）是另一种（更早的）加速 LLM 推理的方法，与 EAGLE 不同。主要区别如下：

 - 草稿生成：使用一个更小、更快的 LLM 作为草稿模型，一次性预测多个后续 token。这与 EAGLE 在特征层面的外推方式截然不同。

 - 验证过程：草稿生成和验证采用链式结构，而 EAGLE 使用基于树的注意力机制。

 - 效率：虽然有效，但通常比 EAGLE 慢。

 - 实现：需要一个独立的草稿模型，对于较小的目标模型来说，找到合适的草稿模型可能比较困难。而 EAGLE 直接改造现有模型架构。

 - 准确性：草稿准确性取决于所用的草稿模型，而 EAGLE 的草稿准确率更高（约 0.8）。

要在 Triton Inference Server 上运行基于草稿模型的投机解码，步骤与上面 EAGLE 的非常相似，唯一区别是需要使用不同的模型仓库。基于草稿模型的投机解码模板位于 [model_repository/opt_model](model_repository/opt_model)，参照 vLLM [文档](https://docs.vllm.ai/en/latest/features/spec_decode.html#speculating-with-a-draft-model)中的示例。复制后按需修改 model.json，然后用以下命令启动 Triton Server：

```bash
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v </path/to/model_repository>:/model_repository \
    nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3 \
    tritonserver --model-repository /model_repository \
    --model-control-mode explicit --load-model opt_model
```
