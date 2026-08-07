
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

# 部署 HuggingFace 模型

**注意**：如果你是 Triton Inference Server 新手，建议先阅读[概念指南第 1 部分](../Conceptual_Guide/Part_1-model_deployment/README.md)。本教程假定你已具备 Triton Inference Server 的基础知识。

|相关页面 | HuggingFace 模型导出指南：[ONNX](https://huggingface.co/docs/transformers/serialization)、[TorchScript](https://huggingface.co/docs/transformers/torchscript) |
| ------------ | --------------- |

开发者经常与开源模型打交道，而 HuggingFace 是众多开源模型的常用来源。本指南将讨论如何用 Triton Inference Server 部署 HuggingFace 上的几乎所有模型。本例使用 [HuggingFace](https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/vit#transformers.ViTModel) 上提供的 [ViT](https://arxiv.org/abs/2010.11929) 模型。

在 Triton Inference Server 上部署模型流水线主要有两种方式：
* **方式 1：** 不把模型从流水线中显式拆分开，直接部署整个流水线。这种方式的核心优势是部署速度快，借助 Triton 的 ["Python 后端"](https://github.com/triton-inference-server/python_backend) 即可实现。更多信息请参考[这个示例](https://github.com/triton-inference-server/python_backend#usage)。简单来说，就是用 Python 后端部署模型/流水线。

* **方式 2：** 拆分流水线：用不同的后端分别处理预处理（preprocessing）和后处理（postprocessing），把核心模型部署在框架后端上。这样做的好处是，核心网络跑在专用框架后端上性能更高，还能利用大量框架特有的优化。更多信息参见概念指南的[第 4 部分](../Conceptual_Guide/Part_4-inference_acceleration/README.md)。这是通过 Triton 的集成（Ensemble）功能实现的，相关说明见概念指南的[第 5 部分](../Conceptual_Guide/Part_5-Model_Ensembles/README.md)。更详细的信息请参阅[文档](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)。简单来说，就是用预处理步骤和导出的模型构建一个 ensemble。

![multiple models](./img/Approach.PNG)

> 💡 **AI Infra 视角**：两种方式的本质区别在性能与灵活性的取舍。方式 1 把整条流水线写进 Python 后端，开发最快，但预处理和模型计算都受 Python 解释开销限制，吞吐上不去；方式 2 把耗时占比高的预处理/后处理放到 Python 后端、核心网络放到 ONNX 等专用后端，并让 Triton 的调度器编排它们。生产实践中的常见路径是：先用方式 1 快速验证效果，再逐步把热点模块拆成方式 2 的架构，让 GPU 利用率真正上去。

## 示例

为了讲解方便，这里使用 `ViT` 模型（[HuggingFace 链接](https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/vit#transformers.ViTModel)）。这个特定的 ViT 模型没有应用头（如图像分类头），不过 [HuggingFace 提供](https://huggingface.co/models?search=google/vit)了带不同头的 ViT 模型供你选用。部署模型时的一个好习惯是：如果不熟悉模型结构，先弄清楚并探索它的结构。使用 [Netron](https://netron.app/) 之类的工具，可以方便地以图形界面查看结构。虽然 Triton 会自动为模型生成配置文件，但构建客户端或模型集成（ensemble）时仍需要输入输出层的名称，这时就可以用这个工具查看。

![multiple models](./img/netron.PNG)

### 在 Python 后端上部署（方式 1）

使用 Triton 的 Python 后端，需要定义 `TritonPythonModel` 类中的最多三个函数：
* `initialize()`：Triton 加载模型时执行此函数。建议用它来初始化/加载模型和数据对象。此函数可选。
```
def initialize(self, args):
    self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
```
* `execute()`：每次请求都会执行此函数，可以在这里放置全部流水线逻辑。
```
def execute(self, requests):
    responses = []
    for request in requests:
        inp = pb_utils.get_input_tensor_by_name(request, "image")
        input_image = np.squeeze(inp.as_numpy()).transpose((2,0,1))
        inputs = self.feature_extractor(images=input_image, return_tensors="pt")

        outputs = self.model(**inputs)

        # Sending results
        inference_response = pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor(
                "label",
                outputs.last_hidden_state.numpy()
            )
        ])
        responses.append(inference_response)
    return responses
```
* `finalize()`：Triton 卸载模型时执行此函数，可用于释放内存或执行卸载模型所需的其他操作。此函数可选。

> 💡 **AI Infra 视角**：这三个函数对应模型的生命周期：`initialize` 只在加载时跑一次，适合做耗时的一次性加载（如从 HuggingFace 拉取权重）；`execute` 是热路径，每个请求都会经过它，其性能直接决定服务吞吐；`finalize` 负责清理。注意 `initialize` 里加载的模型在多个 `execute` 调用之间是共享的——理解这个生命周期，是排查"首次请求特别慢"（模型还没加载完）和显存泄漏问题的起点。

运行这个示例需要两个终端，命令如下：
* **终端 1**：用于启动 Triton Inference Server。
```
# Pick the pre-made model repository
mv python_model_repository model_repository

# Pull and run the Triton container & replace yy.mm
# with year and month of release. Eg. 23.05
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash

# Install dependencies
pip install torch torchvision
pip install transformers
pip install Image

# Launch the server
tritonserver --model-repository=/models
```
* **终端 2**：用于运行客户端。
```
# Pull & run the container
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash

# Run the client

python3 client.py --model_name "python_vit"
```

### 用 Triton 集成（Ensemble）部署（方式 2）

在讨论具体的模型部署细节之前，第一步是下载并导出模型。建议在 [NGC 提供的 PyTorch 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)中执行下面的命令。如果你是第一次在 Triton 中搭建模型集成，强烈建议先阅读[这份指南](../Conceptual_Guide/Part_5-Model_Ensembles/README.md)。拆分流水线的核心优势是性能提升，以及获得大量加速选项。模型加速的细节请参阅概念指南的[第 4 部分](../Conceptual_Guide/Part_4-inference_acceleration/README.md)。

```
# Pull the PyTorch Container from NGC
docker run -it --gpus=all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:26.05-py3

# Install dependencies
pip install transformers
pip install transformers[onnx]

# Export the model
python -m transformers.onnx --model=google/vit-base-patch16-224 --atol=1e-3 onnx/vit
```

模型下载完成后，按下述结构搭建模型仓库。模型仓库的基本结构以及所需的配置文件都放在 `ensemble_model_repository` 中。
```
model_repository/
|-- ensemble_model
|   |-- 1
|   `-- config.pbtxt
|-- preprocessing
|   |-- 1
|   |   `-- model.py
|   `-- config.pbtxt
`-- vit
    `-- 1
        `-- model.onnx
```

> 💡 **AI Infra 视角**：这里的 ensemble 是一张"数据流图"：`ensemble_model/config.pbtxt` 定义各子模型的连接关系，输入先流向 Python 后端的 `preprocessing` 做特征提取，得到 `pixel_values` 再流向 ONNX 后端的 `vit` 模型。Triton 的调度器会像流水线一样编排这些子模型，并把多个子请求自动拼接成批次。拆分出的每个子模型可以独立换后端、独立调优（例如给 vit 开动态批处理），这是生产环境多阶段推理管线的标准组织方式。

这种方式下有三个要点需要考虑。
* **预处理（Preprocessing）**：ViT 的特征提取步骤在 Python 后端上完成。这一步的实现细节与[上面一节](#deploying-on-the-python-backend-approach-1)的过程相同。
* **ViT 模型**：按上面的说明把模型放进仓库即可。Triton Inference Server 会自动生成所需的配置文件。如果想查看生成的配置，可以在启动服务器时加上 `--log-verbose=1`。
* **集成配置（Ensemble Configuration）**：这个配置用来映射集成中两个部分的输入输出层——由 Python 后端处理的 `preprocessing`，以及部署在 ONNX 后端上的 ViT 模型。

运行这个示例与前面流程类似，同样需要两个终端：
* **终端 1**：用于启动 Triton Inference Server。

```
# Pick the pre-made model repository and add the model
mv ensemble_model_repository model_repository
mkdir -p model_repository/vit/1
mv vit/model.onnx model_repository/vit/1/
mkdir model_repository/ensemble_model/1

# Pull and run the Triton container & replace yy.mm
# with year and month of release. Eg. 23.05
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install Image

# Launch the server
tritonserver --model-repository=/models
```
* **终端 2**：用于运行客户端。
```
# Pull & run the container
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash

# Run the client
python3 client.py --model_name "ensemble_model"
```

## 总结

总之，部署大多数 HuggingFace 模型有两种方法：把整个流水线部署在 Python 后端上，或者构建一个集成（ensemble）。

