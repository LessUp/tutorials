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

# 使用 Triton 部署模型（Deploy models using Triton）

| 跳转到 | [第 2 部分：提升资源利用率](../Part_2-improving_resource_utilization/) | [文档：模型仓库（Model Repository）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) | [文档：模型配置（Model Configuration）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md) |
| ------------ | --------------- | --------------- | --------------- |

任何深度学习推理服务方案都需要解决两个根本性挑战：

* 管理多个模型。
* 模型的版本管理、加载与卸载。

## 开始之前

概念指南旨在帮助开发者理解构建深度学习流水线推理基础设施时会遇到的挑战。本系列 `第 1-5 部分` 都在逐步解决同一个简单的问题：部署一条高性能、可扩展的"从图片中转录文本"流水线。这条流水线包含 5 个步骤：

1. 预处理原始图片
2. 检测图片中哪些区域包含文字（文本检测模型）
3. 将图片裁剪到包含文字的区域
4. 计算文字概率（文本识别模型）
5. 将概率转换为实际文本

在 `第 1 部分`，我们先把两个模型部署到 Triton 上，并将前/后处理步骤放在客户端完成。

## 部署多个模型

管理多个模型的核心挑战，是构建一套能满足不同模型不同需求的推理基础设施。例如，用户可能需要在同一台服务器上同时部署 PyTorch 模型和 TensorFlow 模型，两者负载不同、需要运行在不同的硬件设备上，并且需要独立管理各自的 serving 配置（模型队列、版本、缓存、加速等）。Triton Inference Server 能应对以上所有情况，甚至更多。

![multiple models](./img/multiple_models.PNG)

> 💡 **AI Infra 视角**：多个模型共享一个推理服务进程是生产环境的常态——它意味着 GPU 资源可以按负载灵活调配（比如白天 NLP 模型忙、晚上 CV 模型忙），也避免为每个模型单独起服务带来的内存和管理开销。这是推理平台做"多租户"（multi-tenant）的雏形：统一入口、独立配置、共享资源。

使用 Triton Inference Server 部署模型的第一步，是搭建一个存放将要服务的模型及其配置的仓库（repository）。为了演示，我们将使用 [EAST](https://arxiv.org/pdf/1704.03155v2.pdf) 模型做文字检测，以及一个文字识别模型。这个工作流大体改编自 [OpenCV 的文本检测示例](https://docs.opencv.org/4.x/db/da4/samples_2dnn_2text_detection_8cpp-example.html)。

首先，克隆本仓库并进入对应目录。

```bash
cd Conceptual_Guide/Part_1-model_deployment
```

接下来，我们将下载所需的模型，并确保它们处于 Triton 可以部署的格式。

### 模型 1：文本检测

下载并解压 OpenCV 的 EAST 模型。

```bash
wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
tar -xvf frozen_east_text_detection.tar.gz
```

导出为 ONNX 格式。
>注意：以下步骤要求你已安装 TensorFlow 库。我们建议在 NGC TensorFlow 容器环境中执行以下步骤，可用 `docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:<yy.mm>-tf2-py3` 启动该容器。

```bash
pip install -U tf2onnx
python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
```

### 模型 2：文本识别

下载文本识别模型的权重。

```bash
wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth
```

使用 `utils` 文件夹中的模型定义文件将模型导出为 `.onnx` 格式。该文件改编自 [Baek et. al. 2019](https://github.com/clovaai/deep-text-recognition-benchmark)。

>注意：以下 Python 脚本要求你已安装 PyTorch 库。我们建议在 NGC PyTorch 容器环境中执行以下步骤，可用 `docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:<yy.mm>-py3` 启动该容器。

```python
import torch
from utils.model import STRModel

# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load("None-ResNet-None-CTC.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# Create ONNX file by tracing model
trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, trace_input, "str.onnx", verbose=True)
```

### 搭建模型仓库（model repository）

[模型仓库](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)是 Triton 读取模型及每个模型关联元数据（配置、版本文件等）的方式。模型仓库可以存放在本地或网络挂载的文件系统上，也可以放在云对象存储中，如 AWS S3、Azure Blob Storage 或 Google Cloud Storage。关于模型仓库位置的更多细节，请参考[文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-repository-locations)。服务器还可以同时使用多个不同的模型仓库。为简单起见，本讲解只使用一个存放在[本地文件系统](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#local-file-system)的仓库，结构如下：

```bash
# Example repository structure
<model-repository>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
```

> 💡 **AI Infra 视角**：model repository 的目录约定（模型名 / 版本号 / 模型文件）是 Triton 世界里的"部署单元"规范。实际生产中这个目录通常不是手工搭的，而是由 CI/CD 流水线把训练产物（model artifact）按约定结构打包后推到对象存储或挂载到服务器。理解这个结构，是后续做模型发布、灰度、回滚的基础。

上述结构中有一个重要的组成部分需要说明：

* `model-name`：模型的标识名称。
* `config.pbtxt`：用户可以为每个模型定义一份模型配置。这份配置至少需要定义：模型输入输出的 backend、名称、形状和数据类型。对于大多数主流 backend，这份配置文件会根据默认值自动生成。配置文件的完整规范见 [`model_config` protobuf 定义](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)。
* `version`：版本管理使同一模型的多个版本可以按照所选策略被使用。[关于版本管理的更多信息。](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#model-versions)

在本示例中，你可以按如下方式搭建模型仓库结构：

```bash
mkdir -p model_repository/text_detection/1
mv detection.onnx model_repository/text_detection/1/model.onnx

mkdir -p model_repository/text_recognition/1
mv str.onnx model_repository/text_recognition/1/model.onnx
```

这些命令执行后，你的仓库结构应该是这样：

```bash
# Expected folder layout
model_repository/
├── text_detection
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── text_recognition
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

注意，本示例中我们已经创建好 `config.pbtxt` 文件并放到了对应位置。下一节我们将讨论这些文件的内容。

### 模型配置（Model configuration）

模型和文件结构就绪后，下一步需要关注 `config.pbtxt` 模型配置文件。先来看看为你准备好的 `EAST 文本检测` 模型的配置，位于 `/model_repository/text_detection/config.pbtxt`。可以看到 `text_detection` 是一个 ONNX 模型，有 1 个 `input` 和 2 个 `output` 张量。

``` text proto
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 256
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 5 ]
  }
]
```

* `name`："name" 是可选字段，其值应与模型所在目录名一致。
* `backend`：该字段指定用哪个后端运行模型。Triton 支持多种后端，如 TensorFlow、PyTorch、Python、ONNX 等。完整的字段选择列表请参考[这些说明](https://github.com/triton-inference-server/backend#backends)。
* `max_batch_size`：顾名思义，该字段定义模型能支持的最大批大小。
* `input` 和 `output`：输入和输出部分指定名称、形状、数据类型等，同时支持如[重塑（reshaping）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#reshape)和[不规则批处理（ragged batches）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/ragged_batching.md#ragged-batching)等操作。

在[大多数情况下](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration)，可以省略 `input` 和 `output` 部分，让 Triton 直接从模型文件中提取这些信息。这里我们写明了它们，一是为了清晰，二是稍后客户端程序需要知道输出张量的名称。

> 💡 **AI Infra 视角**：`max_batch_size`、`backend`、输入输出定义是 config.pbtxt 里最核心的字段，它们直接决定了 Triton 调度器怎么组合请求。把 `dims` 中为 -1 的维度理解成"可变维度"（允许批处理合并）是理解 Triton 推理能力的关键——后续第 2 部分的动态批处理正是建立在这一基础上。

所有受支持字段及其取值的细节，请参考 [model config protobuf 定义文件](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)。

### 启动服务器

仓库创建好、模型配置完成后，就可以启动服务器了。虽然 Triton Inference Server 可以[从源码构建](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-triton)，但本示例强烈推荐使用 NGC 上免费提供的[预构建 Docker 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)。

```bash
# Replace the yy.mm in the image name with the release year and month
# of the Triton version needed, eg. 22.08

docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3
```

Triton Inference Server 构建完成后，或在容器内部，可以用如下命令启动：

```bash
tritonserver --model-repository=/models
```

这会拉起服务器，模型实例也就绪等待推理了。

```text
I0712 16:37:18.246487 128 server.cc:626]
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| text_detection   | 1       | READY  |
| text_recognition | 1       | READY  |
+------------------+---------+--------+

I0712 16:37:18.267625 128 metrics.cc:650] Collecting metrics for GPU 0: NVIDIA GeForce RTX 3090
I0712 16:37:18.268041 128 tritonserver.cc:2159]
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                                                                        |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                                                                       |
| server_version                   | 2.23.0                                                                                                                                                                                       |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data statistics trace |
| model_repository_path[0]         | /models                                                                                                                                                                                      |
| model_control_mode               | MODE_NONE                                                                                                                                                                                    |
| strict_model_config              | 1                                                                                                                                                                                            |
| rate_limit                       | OFF                                                                                                                                                                                          |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                                                                    |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                                                                                     |
| response_cache_byte_size         | 0                                                                                                                                                                                            |
| min_supported_compute_capability | 6.0                                                                                                                                                                                          |
| strict_readiness                 | 1                                                                                                                                                                                            |
| exit_timeout                     | 30                                                                                                                                                                                           |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

I0712 16:37:18.269464 128 grpc_server.cc:4587] Started GRPCInferenceService at 0.0.0.0:8001
I0712 16:37:18.269956 128 http_server.cc:3303] Started HTTPService at 0.0.0.0:8000
I0712 16:37:18.311686 128 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```

## 构建客户端应用

Triton 服务器启动后，就可以开始向它发送消息了。与 Triton Inference Server 交互有三种方式：

* HTTP(S) API
* gRPC API
* 原生 C API

此外还有预构建的[客户端库](https://github.com/triton-inference-server/client#client-library-apis)，提供 [C++](https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B)、[Python](https://github.com/triton-inference-server/client/tree/main/src/python) 和 [Java](https://github.com/triton-inference-server/client/tree/main/src/java) 版本，它们是对 HTTP 和 gRPC API 的封装。本示例包含一个 Python 客户端脚本 `client.py`，它使用 `tritonclient` Python 库通过 HTTP API 与 Triton 通信。

我们来看看这个文件的内容：

* 首先，我们从 `tritonclient` 库导入 HTTP 客户端，以及处理图片会用到的其他几个库：

  ```python
  import math
  import numpy as np
  import cv2
  import tritonclient.http as httpclient
  ```

* 接下来，我们会定义几个辅助函数，负责流水线的前处理和后处理步骤。为简洁起见这里省略细节，你可以查看 `client.py` 文件了解更多：

  ```python
  def detection_preprocessing(image: cv2.Mat) -> np.ndarray:
    ...

  def detection_postprocessing(scores: np.ndarray, geometry: np.ndarray, preprocessed_image: np.ndarray) -> np.ndarray:
    ...

  def recognition_postprocessing(scores: np.ndarray) -> str:
    ...
  ```

* 然后，我们创建一个客户端对象，并初始化与 Triton Inference Server 的连接：

  ```python
  client = httpclient.InferenceServerClient(url="localhost:8000")
  ```

* 现在，我们根据数据创建要发送给 Triton 的 `InferInput`：

  ```python
  raw_image = cv2.imread("./img2.jpg")
  preprocessed_image = detection_preprocessing(raw_image)

  detection_input = httpclient.InferInput("input_images:0", preprocessed_image.shape, datatype="FP32")
  detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)
  ```

* 最后，我们向 Triton Inference Server 发送推理请求并获取响应：

  ```python
  detection_response = client.infer(model_name="text_detection", inputs=[detection_input])
  ```

* 之后，我们对文本识别模型重复同样的过程：执行下一步处理、创建输入对象、查询服务器，最后做后处理并打印结果。

  ```python
  # Process responses from detection model
  scores = detection_response.as_numpy('feature_fusion/Conv_7/Sigmoid:0')
  geometry = detection_response.as_numpy('feature_fusion/concat_3:0')
  cropped_images = detection_postprocessing(scores, geometry, preprocessed_image)

  # Create input object for recognition model
  recognition_input = httpclient.InferInput("input.1", cropped_images.shape, datatype="FP32")
  recognition_input.set_data_from_numpy(cropped_images, binary_data=True)

  # Query the server
  recognition_response = client.infer(model_name="text_recognition", inputs=[recognition_input])

  # Process response from recognition model
  text = recognition_postprocessing(recognition_response.as_numpy('308'))

  print(text)
  ```

来试一试吧！

```bash
pip install tritonclient[http] opencv-python-headless
python client.py
```

你可能已经注意到，把第一个模型的结果取回来，做一点处理再发回给 Triton，这个过程有些冗余。在本教程的[第 5 部分](../Part_5-Model_Ensembles/)中，我们会探索如何把更多处理步骤移到服务器端，并只用一次网络调用执行多个模型。

## 模型版本管理（Model Versioning）

能部署同一模型的不同版本，对构建 MLOps 流水线至关重要。这个需求来自 A/B 测试、模型版本快速回滚等场景。Triton 用户只需要在同一个仓库里加一个文件夹和新模型即可：

```text
model_repository/
├── text_detection
│   ├── 1
│   │   └── model.onnx
│   ├── 2
│   │   └── model.onnx
│   └── config.pbtxt
└── text_recognition
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

默认情况下 Triton 服务的是"最新"（latest）版本，但服务哪个版本的策略是可定制的。更多信息请[参考这篇指南](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#version-policy)。

> 💡 **AI Infra 视角**：版本管理是 MLOps 中"模型发布"环节的地基。实践中通常的做法是：模型训练完 → 验证 → 将新版本目录放入 model repository → 通过显式加载（EXPLICIT 模式）或版本策略切换流量，实现灰度发布与秒级回滚。一个容易踩的坑：同一模型两个版本并存会同时占用 GPU 显存，做多版本发布前要评估显存预算。

## 模型的加载与卸载（Loading & Unloading Models）

Triton 提供模型管理 API，用于控制模型的加载/卸载策略。当一个或多个模型需要加载或卸载、同时不能中断同一服务器上其他模型的推理时，这个 API 极其有用。用户可以从三种控制模式中选择：

* NONE
* EXPLICIT
* POLL

```bash
tritonserver --model-repository=/models --model-control-mode=poll
```

这些策略也可以在启动服务器时通过命令行参数设置。更多信息请参考文档中的[这一节](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md#model-management)。

# 接下来是什么？

本教程我们覆盖了搭建和查询 Triton Inference Server 的最基础内容。这是 6 部分教程系列的第 1 部分，该系列讨论的是将深度学习模型部署到生产环境所面临的挑战。[第 2 部分](../Part_2-improving_resource_utilization/)讲的是`并发模型执行和动态批处理`。根据你的工作负载和经验，也可以直接跳到[第 5 部分](../Part_5-Model_Ensembles/)，那里讲的是`如何用多个模型、前后处理步骤和业务逻辑构建集成流水线`。
