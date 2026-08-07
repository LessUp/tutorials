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


# 部署 TensorFlow 模型

本教程演示如何在 Triton Inference Server 上部署一个简单的 ResNet 模型。

## 第一步：导出模型

将 TensorFlow 模型导出为 SavedModel 格式。

> 💡 **AI Infra 视角**：不同深度学习框架的部署产物差异明显——PyTorch 对应 torchscript，TensorFlow 对应 SavedModel（内含计算图 `saved_model.pb` 和权重变量），ONNX 则是单一 `.onnx` 文件。Triton 为每个框架提供独立后端（backend），部署时只需把对应产物放进模型仓库并指定后端类型，无需改写模型代码。

```
# <xx.xx> 是 NVIDIA TensorFlow 容器发布标签中的年.月，例如 22.04

docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:<xx.xx>-tf2-py3
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
        +-- model.savedmodel
            |
            +-- saved_model.pb
            +-- variables
                |
                +-- variables.data-00000-of-00001
                +-- variables.index
```

本示例附带的 `config.pbtxt` 是一份模型配置样例。如果你是 Triton 新手，强烈建议先阅读概念指南（conceptual guide）的[第 1 部分](../../Conceptual_Guide/Part_1-model_deployment/README.md)。

```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --backend-config=tensorflow,version=2
```

> 💡 **AI Infra 视角**：启动命令里的 `--backend-config=tensorflow,version=2` 明确告诉 Triton 使用 TensorFlow 2.x 的运行时。Triton 的 TensorFlow 后端同时支持 TF1 与 TF2，两者在算子行为上存在差异，显式指定版本可以避免因默认版本切换导致的推理结果不一致，这在长期维护的生产环境中尤为重要。

## 第三步：用 Triton 客户端查询服务器

安装依赖并下载一张示例图片，用于测试推理。

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk bash
pip install --upgrade tensorflow
pip install image

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

构建客户端需要三步。首先，与 Triton Inference Server 建立连接。

```
triton_client = httpclient.InferenceServerClient(url="localhost:8000")
```

其次，指定模型输入层和输出层的名称。

```
inputs = httpclient.InferInput("input_1", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

output = httpclient.InferRequestedOutput("predictions", binary_data=True, class_count=1000)
```

最后，向 Triton Inference Server 发送推理请求。

```
# Querying the server
results = triton_client.infer(model_name="resnet50", inputs=[inputs], outputs=[output])
predictions = results.as_numpy('predictions')
print(predictions)
```

输出结果大致如下：
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
