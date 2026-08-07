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

# 在 Triton 中部署 Hugging Face Llava1.5-7b 模型

TensorRT-LLM 是 NVIDIA 在 GPU 上运行大语言模型（LLMs）的推荐方案。关于 TensorRT-LLM 的更多信息请参阅[这里](https://github.com/NVIDIA/TensorRT-LLM)，
关于 Triton 的 TensorRT-LLM Backend 请参阅[这里](https://github.com/triton-inference-server/tensorrtllm_backend)。

*注意：* 如果本教程的某些步骤不生效，可能是 `tutorials` 与 `tensorrtllm_backend`
仓库之间存在版本不匹配。必要时请参考 [llama.md](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/llama.md)
了解更详细的修改说明。如果你熟悉 Python，也可以尝试使用
[LLM API](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/README.md)
来驱动 LLM 工作流。


## 获取 Llava1.5-7B 模型

本教程使用带预训练权重的 Llava1.5-7B HuggingFace 模型。请[在此](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)克隆包含权重和分词器的模型仓库。

## 用 Triton Inference Server 部署

接下来将带你走一遍 TensorRT 与 TensorRT-LLM 引擎构建，以及 Triton 模型仓库搭建的完整流程。

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
把 Llava1.5 模型挂载到 `/Llava-1.5-7b-hf`。在 docker 外部建一个 `engines` 文件夹，
以便复用后续运行构建出的引擎。请把 <xx.yy> 替换为你想要使用的 Triton 版本。

```bash
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v </path/to/tensorrtllm_backend>:/tensorrtllm_backend \
    -v </path/to/Llava1.5/repo>:/llava-1.5-7b-hf \
    -v </path/to/engines>:/engines \
    -v </path/to/tutorials>:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-trtllm-python-py3
```

或者，如果你想构建专用容器，可以按照
[这里的说明](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/build.md#build-the-docker-container)
构建带 TensorRT-LLM Backend 的 Triton Server。

启动容器时别忘了允许 GPU 使用。

### 为每个模型构建引擎 [如果已有引擎可跳过此步]

TensorRT-LLM 要求每个模型在运行前先针对你需要的配置完成编译。因此，
在首次于 Triton Server 上运行模型之前，需要先构建一个 TensorRT-LLM 引擎。

从 [24.04 版本](https://github.com/triton-inference-server/server/releases/tag/v2.45.0) 开始，
Triton Server 的 TensorRT-LLM 容器预装了 TensorRT-LLM 包，用户可以直接在 Triton 容器内构建引擎。

Llava1.5 需要两个引擎：一个视觉组件用的 TensorRT 引擎，
和一个语言组件用的 TRT-LLM 引擎。本教程基于 24.05 版本，
对应 TensorRT-LLM 与 TensorRT-LLM backend 的 `v0.9.0`，并遵循
[这个](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.9.0/examples/multimodal#llava-and-vila)
TensorRT-LLM 多模态指南。

> 💡 **AI Infra 视角**：多模态模型（如图文模型）的推理管线由两段组成：视觉编码器（vision encoder，如 CLIP 的 ViT）先把图片切块编码成一组视觉特征向量，再由 LLM 把这些特征作为"视觉 token"混进文本序列做自回归生成。Llava1.5 的两段结构意味着需要分别编译两个引擎：视觉部分用 TensorRT，语言部分用 TRT-LLM。生产部署多模态服务时，视觉编码器的耗时也不可忽视，尤其图片分辨率高时，常需要单独压测和调优这一段的批处理策略。

按以下步骤生成引擎：

```bash
HF_LLAVA_MODEL=/llava-1.5-7b-hf
UNIFIED_CKPT_PATH=/tmp/ckpt/llava/7b/
ENGINE_DIR=/engines/llava1.5
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py
python3 ${CONVERT_CHKPT_SCRIPT} --model_dir ${HF_LLAVA_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
            --output_dir ${ENGINE_DIR} \
            --gemm_plugin float16 \
            --use_fused_mlp \
            --max_batch_size 1 \
            --max_input_len 2048 \
            --max_output_len 512 \
            --max_multimodal_len 576 # 1 (max_batch_size) * 576 (num_visual_features)

python /tensorrtllm_backend/tensorrt_llm/examples/multimodal/build_visual_engine.py --model_path ${HF_LLAVA_MODEL} --model_type llava --output_dir ${ENGINE_DIR}
```

> 💡 **AI Infra 视角**：`--max_multimodal_len` 是为多模态 token 预留的显存预算：Llava1.5 会把每张图片编码成 576 个视觉特征，公式是 `max_batch_size × 576`。理解这类参数的含义才能算清显存账——LLM 的 KV cache 分配上限取决于 `max_input_len + max_output_len + max_multimodal_len` 的总和，三者加起来决定了一个请求最多能消耗多少 KV cache 空间。

> 可选：你可以用同一个 llama 示例目录下的 `run.py` 测试模型的输出。
>
>   ```bash
>    python3 /tensorrtllm_backend/tensorrt_llm/examples/multimodal/run.py --max_new_tokens 30 --hf_model_dir ${HF_LLAVA_MODEL} --visual_engine_dir ${ENGINE_DIR} --llm_engine_dir ${ENGINE_DIR} --decoder_llm --input_text "Question: which city is this? Answer:"
>    ```
> 预期响应如下：
> ```
> [TensorRT-LLM] TensorRT-LLM version: 0.9.0
> ...
> [06/18/2024-01:02:24] [TRT-LLM] [I] ---------------------------------------------------------
> [06/18/2024-01:02:24] [TRT-LLM] [I]
> [Q] Question: which city is this? Answer:
> [06/18/2024-01:02:24] [TRT-LLM] [I]
> [A] ['Singapore']
> [06/18/2024-01:02:24] [TRT-LLM] [I] Generated 1 tokens
> [06/18/2024-01:02:24] [TRT-LLM] [I] ---------------------------------------------------------
> ```

### 用 Triton 提供服务

最后一步是搭建 Triton 模型仓库。本教程已在 `model_repository/` 下提供了所有必要的 Triton 相关文件。
你只需在 `config.pbtxt` 中指定 TensorRT-LLM 引擎的位置：

```bash
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
python3 ${FILL_TEMPLATE_SCRIPT} -i /tutorials/Popular_Models_Guide/Llava1.5/model_repository/tensorrt_llm/config.pbtxt engine_dir:${ENGINE_DIR}
```

3. 启动 Tritonserver

使用 [launch_triton_server.py](https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/scripts/launch_triton_server.py) 脚本。它通过 MPI 启动多个 `tritonserver` 实例。
```bash
export TRT_ENGINE_LOCATION="/engines/llava1.5/visual_encoder.engine"
export HF_LOCATION="/llava-1.5-7b-hf"
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=<world size of the engine> --model_repo=/tutorials/Popular_Models_Guide/Llava1.5/model_repository
```
> 预期响应如下：
> ```
> ...
> I0503 22:01:25.210518 1175 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
> I0503 22:01:25.211612 1175 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
> I0503 22:01:25.254914 1175 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002
> ```

> 💡 **AI Infra 视角**：`TRT_ENGINE_LOCATION` 和 `HF_LOCATION` 这两个环境变量是本教程模型的"接线点"：Python 后端模型（`model.py`）在 `initialize` 阶段读取它们，分别定位视觉编码器引擎和 HuggingFace 模型文件。在多模型共存的模型仓库里，环境变量是向 Python 后端传递配置的常用手段，但要小心多个模型共享同一环境变量时的命名冲突，生产上更推荐通过 `config.pbtxt` 的 `parameters` 传递模型专属配置。

要停止容器内的 Triton Server，运行：
```bash
pkill tritonserver
```

### 发送推理请求

可以用以下方式测试运行结果：
1. [multi_modal_client.py](./multi_modal_client.py) 脚本。

```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials:/tutorials
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

CLIENT_SCRIPT=/tutorials/Popular_Models_Guide/Llava1.5/multi_modal_client.py
python3 ${CLIENT_SCRIPT} --prompt "Describe the picture." --image_url "http://images.cocodataset.org/test2017/000000155781.jpg" --max-tokens=15
```
> 预期响应如下：
> ```
> Got completed request
> The image features a city bus parked on the side of a street.
> ```

2. [generate 端点](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint)。

```bash
curl -X POST localhost:8000/v2/models/llava-1.5/generate -d '{"prompt":"USER: <image>\nQuestion:Describe the picture. Answer:", "image":"http://images.cocodataset.org/test2017/000000155781.jpg", "max_tokens":100}'
```
> 预期响应如下：
> ```
> data: {"completion_tokens":77,"finish_reason":"stop","model_name":"llava-1.5","model_version":"1","prompt_tokens":592,"text":"The image features a city bus parked on the side of a street. The bus is positioned near a railroad crossing, and there is a stop sign visible in the scene. The bus is also displaying an \"Out of Service\" sign, indicating that it is not currently in operation. The street appears to be foggy, adding a sense of atmosphere to the scene.</s>","total_tokens":669}
> ```

## 参考

更多示例请参考 [运行多模态模型的端到端工作流。](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/multimodal/README.md)
