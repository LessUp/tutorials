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

# 部署 ONNX 模型

本教程演示如何在 Triton Inference Server 上部署一个简单的 ResNet 模型。

## 第一步：搭建 Triton Inference Server

使用 Triton 前，需要先构建模型仓库（model repository）。仓库结构如下：
```
model_repository
|
+-- resnet
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.onnx
```

`config.pbtxt` 配置文件是可选的。如果用户不提供，Triton Inference Server 会自动生成配置文件。如果你是 Triton 新手，强烈建议先阅读概念指南（conceptual guide）的[第 1 部分](../../Conceptual_Guide/Part_1-model_deployment/README.md)。

> 💡 **AI Infra 视角**：ONNX 是模型交换的中间表示（intermediate representation），让"训练框架"与"推理后端"解耦——模型可以用 PyTorch 训练、导出为 ONNX，再交给任何支持 ONNX 的推理引擎执行。Triton 能自动推导部分模型的输入输出信息并生成配置，省去手写 `config.pbtxt` 的步骤；但对于形状复杂或带有动态维度的模型，仍建议手动声明配置以精确控制行为。

```
mkdir -p model_repository/densenet_onnx/1
wget -O model_repository/densenet_onnx/1/model.onnx \
    https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx

docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:<yy.mm>-py3 tritonserver --model-repository=/models
```

## Step 3：用 Triton 客户端查询服务器

安装依赖并下载一张示例图片，用于测试推理。

```
docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk bash
pip install torchvision

wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

构建客户端需要三步。首先，与 Triton Inference Server 建立连接。

```
client = httpclient.InferenceServerClient(url="localhost:8000")
```

其次，指定模型输入层和输出层的名称，并描述预期输入的形状（shape）和数据类型（datatype）。

```
inputs = httpclient.InferInput("data_0", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)
```

最后，向 Triton Inference Server 发送推理请求。

```
# Querying the server
results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('fc6_1').astype(str)

print(np.squeeze(inference_output)[:5])
```

输出结果大致如下：
```
['11.549026:92' '11.232335:14' '7.528014:95' '6.923391:17' '6.576575:88']
```

这里的输出格式为 `<置信度分数>:<分类索引>`。关于如何将这些索引映射到类别名称等内容，请参阅我们的[文档](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)。上述客户端代码可在 `client.py` 中找到。
