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

# 在 Triton 中部署 Hugging Face Llama2-7b 模型

TensorRT-LLM 是 NVIDIA 在 GPU 上运行大语言模型（LLMs）的推荐方案。关于 TensorRT-LLM 的更多信息请参阅[这里](https://github.com/NVIDIA/TensorRT-LLM)，
关于 Triton 的 TensorRT-LLM Backend 请参阅[这里](https://github.com/triton-inference-server/tensorrtllm_backend)。

*注意：* 如果本教程的某些步骤不生效，可能是 `tutorials` 与 `tensorrtllm_backend`
仓库之间存在版本不匹配。必要时请参考 [llama.md](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
了解更详细的修改说明。如果你熟悉 Python，也可以尝试使用
[LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md)
来驱动 LLM 工作流。


## 获取 Llama2-7B 模型

本教程使用带预训练权重的 Llama2-7B HuggingFace 模型。请[在此](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)克隆包含权重和分词器的模型仓库。
你需要获得 Llama2 仓库的访问权限，并取得 huggingface cli 的使用权。要获取 huggingface cli 的访问权限，
请访问：[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)。

## 用 Triton CLI 部署

[Triton CLI](https://github.com/triton-inference-server/triton_cli) 是一个开源命令行工具，
支持用户创建、部署和剖析（profile）由 Triton Inference Server 服务的模型。

### 启动 Triton TensorRT-LLM 容器

启动带有 TensorRT-LLM backend 的 Triton Docker 容器。
注意，我们把获取到的 Llama2-7b 模型挂载到容器内的 `/root/.cache/huggingface`，
这样 Triton CLI 可以直接使用它，跳过下载步骤。

在 docker 外部建一个 `engines` 文件夹，以便复用后续运行构建出的引擎。
请把 <xx.yy> 替换为你想要使用的 Triton 版本。

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/Llama2/repo>:/root/.cache/huggingface \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```
### 安装 Triton CLI

安装 [最新版本](https://github.com/triton-inference-server/triton_cli/releases) 的 Triton CLI：
```bash
GIT_REF=<LATEST_RELEASE>
pip install git+https://github.com/triton-inference-server/triton_cli.git@${GIT_REF}
```

### 准备 Triton 模型仓库
Triton CLI 提供一条命令 `triton import`，它会自动把 HF 的 checkpoint 转换成 TensorRT-LLM checkpoint 格式，
构建 TensorRT-LLM 引擎，并准备好 Triton 模型仓库：
```bash
ENGINE_DEST_PATH=/engines triton import -m llama-2-7b --backend tensorrtllm
```

请注意，指定 `ENGINE_DEST_PATH` 是可选的，但如果以后想复用编译好的引擎，建议指定。

`triton import` 成功运行后，控制台会打印出模型仓库的结构：
```
...
triton - INFO - Current repo at /root/models:
models/
├── llama-2-7b/
│   ├── 1/
│   │   ├── lib/
│   │   │   ├── decode.py
│   │   │   └── triton_decoder.py
│   │   └── model.py
│   └── config.pbtxt
├── postprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── preprocessing/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
└── tensorrt_llm/
    ├── 1/
    └── config.pbtxt

```

> 💡 **AI Infra 视角**：Triton CLI 的 `triton import` 把"权重转换 → 引擎构建 → 模型仓库搭建"这三步易错的手工流程压缩成一条命令，适合快速验证想法。但生产中更常见的是把这三步拆成独立的 CI 流水线阶段：权重转换和引擎构建产物（引擎文件）是昂贵的资产，建议缓存并复用，只有模型或配置变更时才重新构建。

### 启动 Triton Inference Server

启动服务器，指向默认模型仓库：
```
triton start
```

### 发送推理请求
使用 [generate 端点](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html)
向已部署的模型发送推理请求。

```bash
curl -X POST localhost:8000/v2/models/llama-2-7b/generate -d '{"text_input": "What is ML?", "max_tokens": 50, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```
> 预期响应如下：
> ```
> {"context_logits":0.0,...,"text_output":"What is ML?\nML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."}
> ```

## 用 Triton Inference Server 部署

如果你希望对部署过程有更强的控制，
接下来将带你走一遍 TensorRT-LLM 引擎构建和 Triton 模型仓库搭建的完整流程。

### 前置条件：TensorRT-LLM backend

本教程需要 TensorRT-LLM Backend 仓库。请注意，
为了获得最佳体验，建议使用 `tensorrtllm_backend` 最新的
[release tag](https://github.com/triton-inference-server/tensorrtllm_backend/tags)，
以及最新的 [Triton Server 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)。

克隆 TensorRT-LLM Backend 仓库，请执行以下命令。
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git  --branch <release branch>
# Update the submodules
cd tensorrtllm_backend
# Install git-lfs if needed
apt-get update && apt-get install git-lfs -y --no-install-recommends
git lfs install
git submodule update --init --recursive
```

### 启动 Triton TensorRT-LLM 容器

启动带有 TensorRT-LLM backend 的 Triton Docker 容器。
注意，为了简便，我们把 `tensorrtllm_backend` 挂载到容器的 `/tensorrtllm_backend`，
把 Llama2 模型挂载到 `/Llama-2-7b-hf`。在 docker 外部建一个 `engines` 文件夹，
以便复用后续运行构建出的引擎。请把 <xx.yy> 替换为你想要使用的 Triton 版本。

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend>:/tensorrtllm_backend \
    -v </path/to/Llama2/repo>:/Llama-2-7b-hf \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

或者，如果你想构建专用容器，可以按照
[这里的说明](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/build.md#build-the-docker-container)
构建带 TensorRT-LLM Backend 的 Triton Server。

启动容器时别忘了允许 GPU 使用。

> 可选：为简便起见，我们把下面的所有步骤浓缩成了一个
> [deploy_trtllm_llama.sh](./deploy_trtllm_llama.sh) 脚本。
> 请先把 tutorials 仓库克隆到你的机器上，并在启动容器时把教程仓库挂载到 `/tutorials`，
> 即在上面的 docker run 命令中加上 `-v /path/to/tutorials/:/tutorials`。
> 容器启动后，直接运行脚本即可：
> ```bash
> /tutorials/Popular_Models_Guide/Llama2/deploy_trtllm_llama.sh <WORLD_SIZE>
> ```
> 如何发送推理请求，请参考本教程的 [发送推理请求](#send-an-inference-request) 一节。

### 为每个模型构建引擎 [如果已有引擎可跳过此步]

TensorRT-LLM 要求每个模型在运行前先针对你需要的配置完成编译。因此，
在首次于 Triton Server 上运行模型之前，需要先构建一个 TensorRT-LLM 引擎。

从 [24.04 版本](https://github.com/triton-inference-server/server/releases/tag/v2.45.0) 开始，
Triton Server 的 TensorRT-LLM 容器预装了 TensorRT-LLM 包，用户可以直接在 Triton 容器内构建引擎。
按以下步骤操作即可：

```bash
HF_LLAMA_MODEL=/Llama-2-7b-hf
UNIFIED_CKPT_PATH=/tmp/ckpt/llama/7b/
ENGINE_DIR=/engines/llama-2-7b/1-gpu/
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${HF_LLAMA_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
            --remove_input_padding enable \
            --gpt_attention_plugin float16 \
            --context_fmha enable \
            --gemm_plugin float16 \
            --output_dir ${ENGINE_DIR} \
            --paged_kv_cache enable \
            --max_batch_size 4
```
> 可选：你可以用同一个 llama 示例目录下的 `run.py` 测试模型的输出。
>
>   ```bash
>    python3 /tensorrtllm_backend/tensorrt_llm/examples/run.py --engine_dir=/engines/llama-2-7b/1-gpu/ --max_output_len 50 --tokenizer_dir /Llama-2-7b-hf --input_text "What is ML?"
>    ```
> 预期响应如下：
> ```
> [TensorRT-LLM] TensorRT-LLM version: 0.17.0.post1
> ...
> Input [Text 0]: "<s> What is ML?"
> Output [Text 0 Beam 0]: "
> ML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."
> ```

### 用 Triton 提供服务

最后一步是创建 Triton 可读取的模型。使用 inflight batching 的模型模板位于
[tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/NVIDIA/TensorRT-LLM/tree/main/triton_backend/all_models/inflight_batcher_llm)。
要运行我们的 Llama2-7B 模型，需要：


1. 复制 inflight batcher 模型仓库

```bash
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
```

2. 修改 preprocessing、postprocessing 和 processing 各阶段的 config.pbtxt。
下面的脚本给出运行 tritonserver 的最小化配置，如果你想追求最佳性能或使用自定义参数，
请阅读[文档](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
和 [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/docs/source/performance/perf-best-practices.md)：
注意：`TRITON_BACKEND` 有两个可选值：`tensorrtllm` 和 `python`。如果 `TRITON_BACKEND=python`，python 后端会部署 [`model.py`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/triton_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/model.py)。
```bash
# preprocessing
TOKENIZER_DIR=/Llama-2-7b-hf/
TOKENIZER_TYPE=auto
ENGINE_DIR=/engines/llama-2-7b/1-gpu/
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRITON_BACKEND=tensorrtllm
LOGITS_DATATYPE="TYPE_FP32"
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}
```

3. 启动 Tritonserver

使用 [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) 脚本。它通过 MPI 启动多个 `tritonserver` 实例。
```bash
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
```
`<world size of the engine>` 是你想用来运行引擎的 GPU 数量。单 GPU 部署设为 1。
> 预期响应如下：
> ```
> ...
> I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
> I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
> I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
> ```

要停止容器内的 Triton Server，运行：
```bash
pkill tritonserver
```
注意：如果因各种原因启动 Tritonserver 失败，别忘了用上面的命令停掉 Triton Server，否则可能引发 OOM 或 MPI 问题。

### 发送推理请求

可以用以下方式测试运行结果：
1. [inflight_batcher_llm_client.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/inflight_batcher_llm/client/inflight_batcher_llm_client.py) 脚本。

```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend/inflight_batcher_llm/client:/tensorrtllm_client \
    -v /path/to/Llama2/repo:/Llama-2-7b-hf \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
# Install extra dependencies for the script
pip3 install transformers sentencepiece
python3 /tensorrtllm_client/inflight_batcher_llm_client.py --request-output-len 50 --tokenizer-dir /Llama-2-7b-hf/ --text "What is ML?"
```
> 预期响应如下：
> ```
> ...
> Input: What is ML?
> Output beam 0:
> ML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation.
> ...
> ```

2. [generate 端点](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint)。

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is ML?", "max_tokens": 50, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```
> 预期响应如下：
> ```
> {"model_name":"ensemble","model_version":"1","sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"What is ML?\nML is a branch of AI that allows computers to learn from data, identify patterns, and make predictions. It is a powerful tool that can be used in a variety of industries, including healthcare, finance, and transportation."}
> ```

### 用 Gen-AI Perf 评估性能
Gen-AI Perf 是一个命令行工具，用于测量经推理服务器服务的生成式 AI 模型的吞吐量与延迟。
关于 Gen-AI Perf 的更多信息请参阅[这里](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/README.html)。

在同一个 Triton Docker 容器（即 nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk）中运行以下命令来使用 Gen-AI Perf：
```bash
genai-perf \
  profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer /Llama-2-7b-hf/ \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```
预期输出类似如下：
```
                                                  LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃              Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│   Request latency (ms) │ 1,630.23 │ 1,616.37 │ 1,644.65 │ 1,644.05 │ 1,638.70 │ 1,635.64 │
│ Output sequence length │   300.00 │   300.00 │   300.00 │   300.00 │   300.00 │   300.00 │
│  Input sequence length │   200.00 │   200.00 │   200.00 │   200.00 │   200.00 │   200.00 │
└────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 184.02
Request throughput (per sec): 0.61
2024-08-08 19:45 [INFO] genai_perf.export_data.json_exporter:56 - Generating artifacts/ensemble-triton-tensorrtllm-concurrency1/profile_export_genai_perf.json
2024-08-08 19:45 [INFO] genai_perf.export_data.csv_exporter:69 - Generating artifacts/ensemble-triton-tensorrtllm-concurrency1/profile_export_genai_perf.csv
```

> 💡 **AI Infra 视角**：LLM 服务性能评估有两个关键维度：输出 token 吞吐（output token throughput，每秒生成 token 数）和请求延迟。注意这里测试的是"合成输入"（synthetic input，长度固定 200 token）——因为 LLM 延迟与输入输出长度强相关，用固定长度的合成负载才能保证压测结果可复现、可比对。真实业务流量长短不齐时，压测结果需要按实际的输入长度分布重新校准。


## 参考

更多示例请参考 [运行 llama 的端到端工作流。](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
