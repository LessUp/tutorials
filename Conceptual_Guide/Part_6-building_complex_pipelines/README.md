<!--
# Copyright 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 构建复杂流水线：Stable Diffusion

| 跳转到 | [第 5 部分：构建模型集成](../Part_5-Model_Ensembles/) | [第 7 部分：迭代调度教程](../Part_7-iterative_scheduling) | [文档：BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting) |
| ------------ | --------------- | --------------- |  --------------- |

**开始本示例之前，请先观看[这个讲解视频](https://youtu.be/JgP2WgNIq_w)了解这条流水线**。本示例重点展示 Triton Inference Server 的两个特性：
* 在同一个推理流水线中使用多个框架。关于受支持框架的更多信息，请[参考这里](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)。
* 使用 Python Backend 的[业务逻辑脚本（Business Logic Scripting）](https://github.com/triton-inference-server/python_backend#business-logic-scripting)API 构建复杂的非线性流水线。

## 使用多个 Backend

构建一条由深度学习模型驱动的流水线是一项协作工作，常常涉及多个贡献者。贡献者往往有不同的开发环境。把不同贡献者的工作拼进同一条流水线时，这可能导致问题。Triton 用户可以使用 Python 或 C++ backend 配合业务逻辑脚本 API（BLS）来触发模型执行，从而解决这个难题。

![Pipeline](./img/multiple_backends.PNG)

在本示例中，模型运行在：
* ONNX Backend
* TensorRT Backend
* Python Backend

部署在框架 backend 上的两个模型都可以用下面的 API 触发：

```
encoding_request = pb_utils.InferenceRequest(
    model_name="text_encoder",
    requested_output_names=["last_hidden_state"],
    inputs=[input_ids_1],
)

response = encoding_request.exec()
text_embeddings = pb_utils.get_output_tensor_by_name(response, "last_hidden_state")
```

> 💡 **AI Infra 视角**：BLS 的本质是在 Triton 内部"再发一次推理请求"——它让 Python backend 里的代码可以像客户端一样调用其他模型。生产中的典型用法：把路由逻辑（根据请求内容决定调哪个模型）、多模型投票、流式处理等放进 BLS，客户端就只看到一个模型。代价是 BLS 脚本是热点路径，要像对待线上服务一样对待它的性能。

完整的示例请参考 `pipeline` 模型中的 `model.py`。

## Stable Diffusion 示例

开始之前，克隆本仓库并进入根目录。为了更好的体验，请使用三个不同的终端。

### 第 1 步：准备服务器环境
* 首先，运行 Triton Inference Server 容器。
```
# Replace yy.mm with year and month of release. Eg. 22.08
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```
* 接下来，安装 python backend 中模型运行所需的所有依赖，并用你的 [huggingface token](https://huggingface.co/settings/tokens) 登录（需要有 [HuggingFace](https://huggingface.co/) 账号）。

```
# PyTorch & Transformers Lib
pip install torch torchvision torchaudio
pip install transformers ftfy scipy accelerate
pip install diffusers==0.9.0
pip install transformers[onnxruntime]
huggingface-cli login
```

### 第 2 步：导出并转换模型
使用 NGC PyTorch 容器导出和转换模型。

```
docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:yy.mm-py3

pip install transformers ftfy scipy
pip install transformers[onnxruntime]
pip install diffusers==0.9.0
huggingface-cli login
cd /mount
python export.py

# Accelerating VAE with TensorRT
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

# Place the models in the model repository
mkdir model_repository/vae/1
mkdir model_repository/text_encoder/1
mv vae.plan model_repository/vae/1/model.plan
mv encoder.onnx model_repository/text_encoder/1/model.onnx
```

### 第 3 步：启动服务器
在服务器容器中启动 Triton Inference Server。

```
tritonserver --model-repository=/models
```

### 第 4 步：运行客户端
使用客户端容器并运行客户端。

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash

# Client with no GUI
python3 client.py

# Client with GUI
pip install gradio packaging
python3 gui/client.py --triton_url="localhost:8001"
```

注意：首次推理查询可能比后续查询耗时更长。

> 💡 **AI Infra 视角**：这个示例是"多引擎流水线"的典型模板：text_encoder 用 ONNX、VAE 用 TensorRT（因为 VAE 是纯卷积网络，TensorRT 加速收益大）、扩散调度循环用 Python BLS 编排。它展示了 Triton 的定位优势——不需要把整条链路都塞进一个框架。做生产部署时，"哪个模型适合哪个引擎"通常要按实测性能决定，而不是一刀切。
