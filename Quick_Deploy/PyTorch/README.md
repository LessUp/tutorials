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


# 部署 PyTorch 模型

本教程演示如何在 Triton Inference Server 上部署一个简单的 ResNet 模型。

## 第一步：导出模型

保存 PyTorch 模型。该模型需要经过追踪（trace）或脚本化（script）处理，以得到 torchscript 格式的模型。

> 💡 **AI Infra 视角**：PyTorch 模型默认是动态计算图，部署前必须序列化为静态图（如 torchscript 或 ONNX），服务端才能脱离训练代码独立加载执行。这一步通常还会固定输入张量的形状，为后续在 GPU 上做算子融合、量化等推理优化打下基础。

```
# <xx.xx> 是 NVIDIA PyTorch 容器发布标签中的年.月，例如 22.04

docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:<xx.xx>-py3
python export.py
```

## 第二步：搭建 Triton Inference Server

使用 Triton 前，需要先构建模型仓库（model repository）。仓库结构如下：
```
model_repository
|
+-- resnet50
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.pt
```

本示例附带的 `config.pbtxt` 是一份模型配置样例。如果你是 Triton 新手，强烈建议先阅读概念指南（conceptual guide）的[第 1 部分](../../Conceptual_Guide/Part_1-model_deployment/README.md)。

> 💡 **AI Infra 视角**：`config.pbtxt` 是 Triton 的模型配置文件，用于声明后端类型、输入输出张量的名称与维度、批处理上限等。每个模型目录下的数字子目录（如 `1`）代表模型版本号，Triton 借此支持多版本共存与无中断的版本切换，这在灰度发布和滚动升级场景中非常实用。

```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## 第三步：用 Triton 客户端查询服务器

安装依赖并下载一张示例图片，用于测试推理。

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk bash
pip install torchvision

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

构建客户端需要三步。首先，与 Triton Inference Server 建立连接。

> 💡 **AI Infra 视角**：Triton 采用客户端-服务端分离架构：服务器通过 HTTP（8000 端口）或 gRPC（8001 端口）对外提供服务，本身不包含任何业务代码；8002 端口则暴露 Prometheus 格式的监控指标。生产环境中客户端通常不会直接面对用户，而是作为推理网关（gateway）的一环，统一接收上层流量再转发给 Triton。

```
client = httpclient.InferenceServerClient(url="localhost:8000")
```

其次，指定模型输入层和输出层的名称。

```
inputs = httpclient.InferInput("input__0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("output__0", binary_data=True, class_count=1000)
```

最后，向 Triton Inference Server 发送推理请求。

```
# Querying the server
results = client.infer(model_name="resnet50", inputs=[inputs], outputs=[outputs])
predictions = results.as_numpy('output__0')
print(predictions[:5])
```

输出结果大致如下：
```
[b'12.468750:90' b'11.523438:92' b'9.664062:14' b'8.429688:136'
 b'8.234375:11']
```

这里的输出格式为 `<置信度分数>:<分类索引>`。关于如何将这些索引映射到类别名称等内容，请参阅我们的[文档](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)。上述客户端代码可在 `client.py` 中找到。
