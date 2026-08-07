<!--
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 使用 Triton Inference Server 部署 Hermes-2-Pro-Llama-3-8B 模型

[Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
是 [NousResearch](https://nousresearch.com/) 开发的先进语言模型。
该模型基于 Meta-Llama-3-8B 微调增强：内部使用 OpenHermes 2.5 数据集微调，
并新增了 NousResearch 自研的函数调用（Function Calling）与 JSON 模式（JSON Mode）数据集。
这些改进使模型既能胜任通用对话任务，也擅长结构化 JSON 输出、函数调用等专项功能，
成为适用于多种应用场景的多面手。

> 💡 **AI Infra 视角**：函数调用与 JSON 模式是当前 LLM 应用落地的关键能力——Agent 系统靠函数调用让模型去调用工具，业务系统靠 JSON 模式拿到结构化输出。从服务端角度看，这类能力不改变推理引擎本身，模型仍是标准的自回归生成，区别只在于提示词模板与解码策略，因此可以直接复用现成的推理管线，这也是本教程能沿用标准 TRT-LLM 部署流程的原因。

该模型可通过 [huggingface](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 下载。

TensorRT-LLM 是 NVIDIA 在 GPU 上运行大语言模型（LLMs）的推荐方案。关于 TensorRT-LLM 的更多信息请参阅[这里](https://github.com/NVIDIA/TensorRT-LLM)，
关于 Triton 的 TensorRT-LLM Backend 请参阅[这里](https://github.com/triton-inference-server/tensorrtllm_backend)。

*注意：* 如果本教程的某些步骤不生效，可能是 `tutorials` 与 `tensorrtllm_backend`
仓库之间存在版本不匹配。必要时请参考 [llama.md](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
了解更详细的修改说明。如果你熟悉 Python，也可以尝试使用
[LLM API](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llm-api/README.md)
来驱动 LLM 工作流。

## 前置条件：TensorRT-LLM backend

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

## 启动 Triton TensorRT-LLM 容器

启动带有 TensorRT-LLM backend 的 Triton Docker 容器。
注意，为了简便，我们把 `tensorrtllm_backend` 挂载到容器的 `/tensorrtllm_backend`，
把 Hermes 模型挂载到 `/Hermes-2-Pro-Llama-3-8B`。在 docker 外部建一个 `engines` 文件夹，
以便复用后续运行构建出的引擎。请把 <xx.yy> 替换为你想要使用的 Triton 版本。

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend>:/tensorrtllm_backend \
    -v </path/to/Hermes/repo>:/Hermes-2-Pro-Llama-3-8B \
    -v </path/to/engines>:/engines \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

或者，如果你想构建专用容器，可以按照
[这里的说明](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/build.md#build-the-docker-container)
构建带 TensorRT-LLM Backend 的 Triton Server。

启动容器时别忘了允许 GPU 使用。

## 为每个模型构建引擎 [如果已有引擎可跳过此步]

TensorRT-LLM 要求每个模型在运行前先针对你需要的配置完成编译。因此，
在首次于 Triton Server 上运行模型之前，需要先构建一个 TensorRT-LLM 引擎。

Triton Server 的 TensorRT-LLM 容器预装了 TensorRT-LLM 包，用户可以直接在 Triton 容器内构建引擎。
按以下步骤操作即可：

```bash
HF_LLAMA_MODEL=/Hermes-2-Pro-Llama-3-8B
UNIFIED_CKPT_PATH=/tmp/ckpt/hermes/8b/
ENGINE_DIR=/engines
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
>    python3 /tensorrtllm_backend/tensorrt_llm/examples/run.py --engine_dir=${ENGINE_DIR} --max_output_len 28 --tokenizer_dir ${HF_LLAMA_MODEL} --input_text "What is ML?"
>    ```
> 预期响应如下：
> ```
> Input [Text 0]: "<|begin_of_text|>What is ML?"
> Output [Text 0 Beam 0]: "
> Machine learning is a type of artificial intelligence (AI) that allows software applications to become more accurate in predicting outcomes without being explicitly programmed."
> ```

> 💡 **AI Infra 视角**：TRT-LLM 的构建流程分两步：先把 HuggingFace 的权重（checkpoint）转换成 TRT-LLM 的统一格式（unified checkpoint），再用 `trtllm-build` 编译成引擎。其中 `--gpt_attention_plugin`、`--gemm_plugin` 等开关决定算子是否走高度优化的融合实现，`--paged_kv_cache enable` 开启分页 KV cache 以支撑高并发，`--max_batch_size` 上限直接决定能同时服务的请求数。这些编译期开关在生产中要反复调优，通常建议先用默认配置跑通，再用 profiling 结果逐步开启优化项。

## 用 Triton 提供服务

最后一步是创建 Triton 可读取的模型。使用 inflight batching 的模型模板位于
[tensorrtllm_backend/all_models/inflight_batcher_llm](https://github.com/NVIDIA/TensorRT-LLM/tree/main/triton_backend/all_models/inflight_batcher_llm)。
要运行我们的模型，需要：


1. 复制 inflight batcher 模型仓库

```bash
cp -R /tensorrtllm_backend/all_models/inflight_batcher_llm /opt/tritonserver/.
```

2. 修改 preprocessing、postprocessing 和 processing 各阶段的 `config.pbtxt`。
下面的脚本给出运行 tritonserver 的最小化配置，如果你想追求最佳性能或使用自定义参数，
请阅读[文档](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
和 [perf_best_practices](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/docs/source/performance/perf-best-practices.md)：

```bash
# preprocessing
TOKENIZER_DIR=/Hermes-2-Pro-Llama-3-8B/
TOKENIZER_TYPE=auto
DECOUPLED_MODE=false
MODEL_FOLDER=/opt/tritonserver/inflight_batcher_llm
MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=10000
TRTLLM_BACKEND=python
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRTLLM_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching
```

> 💡 **AI Infra 视角**：一个完整的 TRT-LLM 服务由多个 Triton 模型组合而成：preprocessing（分词）、tensorrt_llm（引擎推理）、postprocessing（解码）、tensorrt_llm_bls（业务逻辑编排）以及 ensemble（把它们串成一条管线）。inflight batching 是 TRT-LLM 的持续批处理实现：解码阶段请求按 token 粒度动态进出批次，新请求能插入到正在执行的批次中（所以叫 "inflight"，飞行中），从而把 GPU 利用率拉到接近满载。

3. 启动 Tritonserver

> [!NOTE]
> 本教程面向单 GPU 上部署 TensorRT-LLM 模型。因此，如果引擎是单 GPU 构建的，
> 下面的命令中使用 `--world_size=1`。如果引擎需要多 GPU，务必在 `--world_size` 中
> 指定引擎所需的确切 GPU 数量。

使用 [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) 脚本。它通过 MPI 启动多个 `tritonserver` 实例。
```bash
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/opt/tritonserver/inflight_batcher_llm
```
> 预期响应如下：
> ```
> ...
> I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
> I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
> I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
> ```

> 💡 **AI Infra 视角**：`--world_size` 对应模型并行（model parallelism）中的张量并行（tensor parallelism）卡数：引擎按 N 卡编译后，权重和计算会切分到 N 张 GPU 上，每层计算通过 NVLink/PCIe 做集合通信（all-reduce）汇总。world_size 必须在构建引擎和启动服务两处保持一致，否则会报错。多卡能装下更大的模型，但通信开销会拉低单 token 延迟——8B 这类小模型单卡即可，70B+ 才需要多卡。

要停止容器内的 Triton Server，运行：
```bash
pkill tritonserver
```

## 发送推理请求

可以用以下方式测试运行结果：
1. [inflight_batcher_llm_client.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/inflight_batcher_llm/client/inflight_batcher_llm_client.py) 脚本。

首先启动 Triton SDK 容器：
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tensorrtllm_backend/inflight_batcher_llm/client:/tensorrtllm_client \
    -v /path/to/Hermes-2-Pro-Llama-3-8B/repo:/Hermes-2-Pro-Llama-3-8B \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

另外，请为脚本安装额外的依赖：
```bash
pip3 install transformers sentencepiece
python3 /tensorrtllm_client/inflight_batcher_llm_client.py --request-output-len 28 --tokenizer-dir /Hermes-2-Pro-Llama-3-8B --text "What is ML?"
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
> {"context_logits":0.0,...,"text_output":"What is ML?\nMachine learning is a type of artificial intelligence (AI) that allows software applications to become more accurate in predicting outcomes without being explicitly programmed."}
> ```


## 参考

更多示例请参考 [运行 llama 的端到端工作流。](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
