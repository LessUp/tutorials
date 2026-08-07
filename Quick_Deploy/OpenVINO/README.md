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

# 用 OpenVINO 后端部署 ONNX、PyTorch 和 TensorFlow 模型

本教程演示如何使用 [OpenVINO 后端](https://github.com/triton-inference-server/openvino_backend)，在 Triton Inference Server 上部署简单的 ONNX、PyTorch 和 TensorFlow 模型。

> 💡 **AI Infra 视角**：OpenVINO 是 Intel 出品的推理引擎，擅长在 Intel CPU、集成显卡（iGPU）和 VPU 等非 NVIDIA 硬件上高效运行模型。在实际的推理集群里，GPU 通常留给最重的模型，CPU 上跑 OpenVINO 既可以让存量 Intel 服务器继续发挥价值，也能为 GPU 腾出算力——理解它能部署什么、不能部署什么，有助于做好异构资源规划。

## 部署 ONNX 模型
### 1. 构建模型仓库并下载 ONNX 模型。
```
mkdir -p model_repository/densenet_onnx/1
wget -O model_repository/densenet_onnx/1/model.onnx \
    https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx
```

### 2. 新建一个名为 `config.pbtxt` 的文件
```
name: "densenet_onnx"
backend: "openvino"
default_model_filename: "model.onnx"
```

### 3. 把 `config.pbtxt` 放入模型仓库，最终结构应如下所示：
```
model_repository
|
+-- densenet_onnx
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.onnx
```

注意：Triton Inference Server 正是依靠这种目录结构来读取配置和模型文件，必须遵循规定的布局。除模型必需的文件外，不要在模型仓库中放置任何其他文件夹或文件。

### 4. 运行 Triton Inference Server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:26.07-py3 tritonserver --model-repository=/models
```

### 5. 从 GitHub 下载 Triton 客户端代码 `client.py` 到你想要运行客户端的位置。
```
wget https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/ONNX/client.py
```

### 6. 在 `client.py` 所在位置运行 Triton 客户端，安装依赖并查询服务器
构建客户端需要三步：首先，与 Triton Inference Server 建立连接；其次，指定模型输入层和输出层的名称；最后，向 Triton Inference Server 发送推理请求。
```
docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:26.07-py3-sdk bash
```
```
pip install torchvision
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
python3 client.py
```

### 7. 输出
```
['11.549026:92' '11.232335:14' '7.528014:95' '6.923391:17' '6.576575:88']
```
这里的输出格式为 `<置信度分数>:<分类索引>`。关于如何将这些索引映射到类别名称等内容，请参阅我们的[文档](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)。上述客户端代码可在 `client.py` 中找到。

## 部署 PyTorch 模型
### 1. 下载并准备 PyTorch 模型。
PyTorch 模型（.pt）需要转换为 OpenVINO 格式。创建一个 `downloadAndConvert.py` 文件，下载 PyTorch 模型并用 OpenVINO 模型转换器（Model Converter）保存出 `model.xml` 和 `model.bin`：
```
import torchvision
import torch
import openvino as ov
model = torchvision.models.resnet50(weights='DEFAULT')
ov_model = ov.convert_model(model)
ov.save_model(ov_model, 'model.xml')
```

> 💡 **AI Infra 视角**：OpenVINO 的模型产物是 IR（Intermediate Representation）格式，由描述计算图的 `model.xml` 和存放权重的 `model.bin` 两部分组成。与 TensorRT 的引擎文件类似，IR 是经过转换器优化后的中间表示，运行时再针对具体硬件生成可执行代码——转换一次、多端部署，这正是异构推理平台的常见做法。

安装依赖：
```
pip install openvino
pip install torchvision
```

运行 `downloadAndConvert.py`
```
python3 downloadAndConvert.py
```

要转换你自己的 PyTorch 模型，请参阅[转换 PyTorch 模型](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-pytorch.html)

### 2. 新建一个名为 `config.pbtxt` 的文件
```
name: "resnet50 "
backend: "openvino"
max_batch_size : 0
input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]
output [
  {
    name: "x.45"
    data_type: TYPE_FP32
    dims: [ 1, 1000 ,1, 1]
    reshape { shape: [ 1, 1000 ] }
  }
]
```

3. 把 `config.pbtxt` 以及 `model.xml`、`model.bin` 放入模型仓库，文件夹结构应如下所示：
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.xml
        +-- model.bin
```

注意：Triton Inference Server 正是依靠这种目录结构来读取配置和模型文件，必须遵循规定的布局。除模型必需的文件外，不要在模型仓库中放置任何其他文件夹或文件。

### 4. 运行 Triton Inference Server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:26.07-py3 tritonserver --model-repository=/models
```

### 5. 在另一个终端，从 GitHub 下载 Triton 客户端代码 `client.py` 到你想运行客户端的位置。
```
wget https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/PyTorch/client.py
```

由于该模型与 Triton 教程中的模型略有不同，你需要在 `client.py` 中把模型的输入输出名称改为后端期望的名称。例如，把 PyTorch 模型原本的输入名（input__0）改为 OpenVINO 后端使用的名称（x）。

| Old Value   | New Value |
| :-------------: | :-------------: |
| input__0 |x |
| output__0 |x.45 |

> 💡 **AI Infra 视角**：同一个模型在不同后端下，输入输出张量的命名可能完全不同（如 `input__0` vs `x`）。因此部署新后端时，首要任务是确认张量名——用 Netron 等工具打开模型文件查看即可。命名不匹配是推理服务对接中最常见的低级错误，客户端、ensemble 配置和 `config.pbtxt` 三处必须保持一致。

### 6. 在 `client.py` 所在位置运行 Triton 客户端，安装依赖并查询服务器。
构建客户端需要三步：首先，与 Triton Inference Server 建立连接；其次，指定模型输入层和输出层的名称；最后，向 Triton Inference Server 发送推理请求。
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:26.07-py3-sdk bash
```
```
pip install torchvision
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
python3 client.py
```

### 7. 输出
```
[b'6.354599:14' b'4.292510:92' b'3.886345:90' b'3.333909:136' b'3.096908:15']
```
这里的输出格式为 `<置信度分数>:<分类索引>`。关于如何将这些索引映射到类别名称等内容，请参阅我们的[文档](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)。上述客户端代码可在 `client.py` 中找到。


## 部署 TensorFlow 模型
### 1. 下载并准备 TensorFlow 模型。
以 SavedModel 格式导出 TensorFlow 模型：
```
docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:26.05-tf2-py3
```
```
python3 export.py
```

模型需要转换为 OpenVINO 格式。创建一个 `convert.py` 文件，用 OpenVINO 模型转换器保存出 `model.xml` 和 `model.bin`：
```
import openvino as ov
ov_model = ov.convert_model(' path_to_saved_model_dir’)
ov.save_model(ov_model, 'model.xml')
```

安装依赖：
```
pip install openvino
```

运行 `convert.py`
```
python3 convert.py
```

要转换你的 TensorFlow 模型，请参阅[转换 TensorFlow 模型](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-tensorflow.html)

### 2. 新建一个名为 `config.pbtxt` 的文件
```pbtxt
name: "resnet50"
backend: "openvino"
max_batch_size : 0
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    dims: [-1, 224, 224, 3 ]
  }
]
output [
  {
    name: "predictions"
    data_type: TYPE_FP32
    dims: [-1, 1000]
  }
]
```

### 3. 把 `config.pbtxt` 以及 `model.xml`、`model.bin` 放入模型仓库，结构应如下所示：
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.xml
        +-- model.bin
```

注意：Triton Inference Server 正是依靠这种目录结构来读取配置和模型文件，必须遵循规定的布局。除模型必需的文件外，不要在模型仓库中放置任何其他文件夹或文件。

### 4. 运行 Triton Inference Server
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:26.07-py3 tritonserver --model-repository=/models
```

### 5. 在另一个终端，从 GitHub 下载 Triton 客户端代码 `client.py` 到你想运行客户端的位置。
```
wget https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/TensorFlow/client.py
```

### 6. 在 `client.py` 所在位置运行 Triton 客户端，安装依赖并查询服务器。
构建客户端需要三步：首先，与 Triton Inference Server 建立连接；其次，指定模型输入层和输出层的名称；最后，向 Triton Inference Server 发送推理请求。
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:26.07-py3-sdk bash
```
```
pip install --upgrade tensorflow
pip install image
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
python3 client.py
```

### 7. 输出
```
[b'0.301167:90' b'0.169790:14' b'0.161309:92' b'0.093105:94'

 b'0.058743:136' b'0.050185:11' b'0.033802:91' b'0.011760:88'

 b'0.008309:989' b'0.004927:95' b'0.004905:13' b'0.004095:317'

 b'0.004006:96' b'0.003694:12' b'0.003526:42' b'0.003390:313'

 ...

 b'0.000001:751' b'0.000001:685' b'0.000001:408' b'0.000001:116'

 b'0.000001:627' b'0.000001:933' b'0.000000:661' b'0.000000:148']
```
这里的输出格式为 `<置信度分数>:<分类索引>`。关于如何将这些索引映射到类别名称等内容，请参阅我们的[文档](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)。上述客户端代码可在 `client.py` 中找到。
