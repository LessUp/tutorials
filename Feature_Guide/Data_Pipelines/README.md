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

# 理解数据管线（Data Pipelines）

学会把张量（tensor）从客户端搬运到 backend，以及在 backend 之间搬运，是 Triton 用户必须掌握的核心技能。本教程将涵盖以下内容：

* 用于在客户端与服务器之间搬运数据的 API
* 如何熟练使用 ensemble 调度器（ensemble scheduler）

**注意：** 本示例假设读者对 Triton Inference Server 的基本用法已有了解。如果你是 Triton Inference Server 新手，请先阅读[概念指南的第 1 部分](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment)再继续。

> 💡 **AI Infra 视角**：数据管线（Data Pipeline）解决的是推理服务中「请求进来后如何流转」的问题。在 LLM 场景里，一个典型管线可能是「文本预处理 → tokenizer → LLM → 后处理 → 返回」，多模型服务则可能是「检测模型 → 分类模型」串联。Triton 的 ensemble 调度器把管线编排下沉到服务端：数据在模型之间的搬运（CPU/GPU 内存拷贝）由 Triton 统一调度，客户端只需发一次请求，省去了多次网络往返，也避免了把中间结果从 GPU 搬回 CPU 再传回的浪费。

## 伪管线（dummy pipeline）概览

要搬运的数据类型取决于你构建的管线类型，因此不存在一个能覆盖所有人群的通用真实示例。本教程将简单演示如何通过一个伪管线搬运 String、UINT8 与 INT8 数组、FP32 图像和布尔值。

<p align="center" width="100%">
    <img src="./img/Flow.PNG" width="75%">
</p>

### 设置模型与 Ensemble

在继续之前，先设置模型。为了演示，我们使用一种 ["Python 模型"](https://github.com/triton-inference-server/python_backend#python-backend)。Triton 中的 Python 模型本质上是一个包含三个 Triton 特有函数的类：`initialize`、`execute` 和 `finalize`。用户可以在 Python 运行时中加载任何自己编写的 Python 函数或模型来定制这个类。`initialize` 函数在 Python 模型加载到内存时运行，`finalize` 函数在模型从内存卸载时运行。这两个函数都是可选的。同样，为了简单起见，本示例只使用 `execute` 函数来打印"Python 模型"收到的张量。看看是怎么做的：

```
def execute(self, requests):
    responses = []
    for request in requests:
        inp = pb_utils.get_input_tensor_by_name(request, "model_1_input_string")
        inp2 = pb_utils.get_input_tensor_by_name(request, "model_1_input_UINT8_array")
        inp3 = pb_utils.get_input_tensor_by_name(request, "model_1_input_INT8_array")
        inp4 = pb_utils.get_input_tensor_by_name(request, "model_1_input_FP32_image")
        inp5 = pb_utils.get_input_tensor_by_name(request, "model_1_input_bool")

        print("Model 1 received", flush=True)
        print(inp.as_numpy(), flush=True)
        print(inp2.as_numpy(), flush=True)
        print(inp3.as_numpy(), flush=True)
        print(inp4.as_numpy(), flush=True)
        print(inp5.as_numpy(), flush=True)

        inference_response = pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor(
                "model_1_output_string",
                inp.as_numpy(),
            ),
            pb_utils.Tensor(
                "model_1_output_UINT8_array",
                inp2.as_numpy(),
            ),
            pb_utils.Tensor(
                "model_1_output_INT8_array",
                inp3.as_numpy(),
            ),
            pb_utils.Tensor(
                "model_1_output_FP32_image",
                inp4.as_numpy(),
            ),
            pb_utils.Tensor(
                "model_1_output_bool",
                inp5.as_numpy(),
            )
        ])
        responses.append(inference_response)
    return responses
```

这里有两个关键点：`pb_utils.get_input_tensor_by_name(...)` 和 `pb_utils.InferenceResponse(...)` 函数。顾名思义，这两个函数分别用于接收和发送张量。Triton Inference Server 支持多种数据类型。本示例展示了其中 5 种，完整的数据类型列表参见[文档](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes)。

> 💡 **AI Infra 视角**：`pb_utils.get_input_tensor_by_name` 和 `pb_utils.Tensor` 是 Python backend 里读写张量的标准姿势：前者拿到的是引用，后者负责把 NumPy 数组包装成输出张量。注意 Triton 的张量搬运是「按名字约定」的——输入输出张量名必须在 `config.pbtxt` 中声明，名字不匹配的请求会在入口被拒。生产上最常见的排障问题之一就是 client、model config、model.py 三处的名字/数据类型不一致。

在本模型中，"输入层"是 `model_1_input_string`、`model_1_input_UINT8_array`、`model_1_input_INT8_array`、`model_1_input_FP32_image` 和 `model_1_input_bool`。我们在这个模型的 `config.pbtxt` 中连同预期的维度和数据类型一起定义它们。
```
input [
  {
    name: "model_1_input_string"
    data_type: TYPE_STRING
    dims: [-1]
  },
  {
    name: "model_1_input_UINT8_array"
    data_type: TYPE_UINT8
    dims: [-1]
  },
  {
    name: "model_1_input_INT8_array"
    data_type: TYPE_INT8
    dims: [-1]
  },
  {
    name: "model_1_input_FP32_image"
    data_type: TYPE_FP32
    dims: [-1, -1, -1]
  },
  {
    name: "model_1_input_bool"
    data_type: TYPE_BOOL
    dims: [-1]
  }
]
```

类似地，"输出层"是 `model_1_output_string`、`model_1_output_UINT8_array`、`model_1_output_INT8_array`、`model_1_output_FP32_image` 和 `model_1_output_bool`，在 `config.pbtxt` 中定义如下：

```
output [
  {
    name: "model_1_output_string"
    data_type: TYPE_STRING
    dims: [-1]
  },
  {
    name: "model_1_output_UINT8_array"
    data_type: TYPE_UINT8
    dims: [-1]
  },
  {
    name: "model_1_output_INT8_array"
    data_type: TYPE_INT8
    dims: [-1]
  },
  {
    name: "model_1_output_FP32_image"
    data_type: TYPE_FP32
    dims: [-1, -1, -1]
  },
  {
    name: "model_1_output_bool"
    data_type: TYPE_BOOL
    dims: [-1]
  }
]
```

**注意**：对于普通的 `onnx`、`torchscript`、`tensorflow` 或其他模型，我们只需在 `config.pbtxt` 中定义输入输出层即可。ensemble 与客户端之间的交互方式保持不变。如果你不确定模型的层、维度和数据类型，可以使用 [Netron](https://netron.app/) 或 [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) 等工具获取所需信息。

本示例中的第二个模型与上面完全一样。我们将用它来展示[模型 ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)中的数据流。如果你已经读过[概念指南的第 5 部分](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_5-Model_Ensembles)，下面的 ensemble 解释可能看起来很熟悉。

模型设置好之后，我们来讨论如何搭建 ensemble。Ensemble 用于构建由两个或更多模型组成的管线。使用 ensemble 的好处在于，Triton Inference Server 会处理两个模型之间所需的全部张量/内存搬运。此外，用户只需用简单的配置文件即可定义模型流程。当用户需要搭建多条管线、且其中共享一些公共模型时，这一特性尤为有用。

我们稍后再讨论模型仓库的结构，先来看 ensemble 的配置。

由于所有张量的流转方式相同，我们只关注字符串输入。ensemble 模型的完整配置大致如下：

```
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 8
input [
  {
    name: "ensemble_input_string"
    data_type: TYPE_STRING
    dims: [-1]
  },
  ...
]
output [
  {
    name: "ensemble_output_string"
    data_type: TYPE_STRING
    dims: [-1]
  },
  ...
]

ensemble_scheduling {
  step [
    {
      model_name: "model1"
      model_version: -1
      input_map {
        key: "model_1_input_string"
        value: "ensemble_input_string"
      },
      ...

      output_map {
        key: "model_1_output_string"
        value: "model1_to_model2_string"
      },
      ...

    },
    {
      model_name: "model2"
      model_version: -1
      input_map {
        key: "model_2_input_string"
        value: "model1_to_model2_string"
      },
      ...

      output_map {
        key: "model_2_output_string"
        value: "ensemble_output_string"
      },
      ...

    }
  ]
```

我们来拆解一下：首先定义整个 ensemble 的输入和输出。
```
input [
  {
    name: "ensemble_input_string"
    data_type: TYPE_STRING
    dims: [-1]
  },
  ...
]
output [
  {
    name: "ensemble_output_string"
    data_type: TYPE_STRING
    dims: [-1]
  },
  ...
]
```
这与在普通模型中定义输入输出层类似。接下来定义 ensemble 的具体流程。流程由"步骤"（steps）组成，每个步骤定义该步骤要执行的模型及其输入输出。
```
ensemble_scheduling {
  step [
    {
      model_name: "model1"
      model_version: -1
      ...

    },
    {
      model_name: "model2"
      model_version: -1
      ...

    }
  ]
```
用户首先需要理解的是如何定义 ensemble 管线的整体流程。例如，哪个模型先运行？然后，张量如何在每个模型/步骤之间流转？为此，我们使用 `input_map` 和 `output_map`。

```
ensemble_scheduling {
  step [
    {
      model_name: "model1"
      model_version: -1
      input_map {
        key: "model_1_input_string"       # Model 1's input Tensor
        value: "ensemble_input_string"    # this is the name of the ensemble's input
      },
      ...

      output_map {
        key: "model_1_output_string"      # Model 1's output Tensor
        value: "model1_to_model2_string"  # Mapping output from Model1 to Model2
      },
      ...

    },
    {
      model_name: "model2"
      model_version: -1
      input_map {
        key: "model_2_input_string"       # Model 2's input Tensor
        value: "model1_to_model2_string"  # Mapping output from Model1 to Model2
      },
      ...

      output_map {
        key: "model_2_output_string"      # Model 2's output Tensor
        value: "ensemble_output_string"   # this is the name of the ensemble's output
      },
      ...

    }
  ]
```
继续之前，先建立对 `key` 和 `value` 字段的理解。`key` 字段填的是模型所需的层名。`value` 字段由 ensemble 识别，用于定义张量的流转方向。所以，如果想把某个模型某个输出层的输出传给另一个模型的输入，就需要把 `model1` 的 `output_map` 中的 `value` 用作 `model2` 的 `input_map` 中的 `value`。

> 💡 **AI Infra 视角**：`input_map` / `output_map` 本质上是一张「管线拓扑图」：`key` 是子模型自己的张量名，`value` 是 ensemble 层的"总线名"。多个步骤通过共享 `value` 隐式完成张量衔接，Triton 会在 GPU 上直接传递张量（无需经过 CPU 或磁盘），因此多模型串联时尽量不要在中间插入不必要的拷贝。另外 `max_batch_size: 8` 表示 ensemble 会按 batch 维度动态拼接请求——每个子模型依然独立调度，ensemble 只是定义了数据依赖。

弄清了各部分的配置，再简单看一下本示例模型仓库的结构。本质上我们有两个模型：
```
model_repository/
├── ensemble_model
│   ├── 1               # Empty version folder required for ensemble models
│   └── config.pbtxt    # Config for the Ensemble
├── model1
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt    # Config for model 1
└── model2
    ├── 1
    │   └── model.py
    └── config.pbtxt    # Config for model 2
```

### 理解 Python 客户端

服务器端配置完成后，我们来讨论客户端代码。
```
def main():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    prompts = ["This is a string"]
    text_obj = np.array([prompts], dtype="object")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = np.asarray(Image.open(requests.get(url, stream=True).raw)).astype(np.float32)
    uint8_array = np.expand_dims(np.array([1,2,3], dtype = np.uint8), axis = 0)
    int8_array = np.expand_dims(np.array([-1,2,-3], dtype = np.int8), axis = 0)
    image = np.expand_dims(image, axis=0)
    boolean = np.expand_dims(np.array([True]), axis = 0)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("ensemble_input_string", text_obj.shape,np_to_triton_dtype(text_obj.dtype)),
        httpclient.InferInput("ensemble_input_UINT8_array", uint8_array.shape, datatype="UINT8"),
        httpclient.InferInput("ensemble_input_INT8_array", int8_array.shape, datatype="INT8"),
        httpclient.InferInput("ensemble_input_FP32_image", image.shape, datatype="FP32"),
        httpclient.InferInput("ensemble_input_bool", boolean.shape, datatype="BOOL")
    ]
    input_tensors[0].set_data_from_numpy(text_obj)
    input_tensors[1].set_data_from_numpy(uint8_array)
    input_tensors[2].set_data_from_numpy(int8_array)
    input_tensors[3].set_data_from_numpy(image)
    input_tensors[4].set_data_from_numpy(boolean)

    # Set outputs
    output = [
        httpclient.InferRequestedOutput("ensemble_output_string"),
        httpclient.InferRequestedOutput("ensemble_output_UINT8_array"),
        httpclient.InferRequestedOutput("ensemble_output_INT8_array"),
        httpclient.InferRequestedOutput("ensemble_output_FP32_image"),
        httpclient.InferRequestedOutput("ensemble_output_bool")
    ]

    # Query
    query_response = client.infer(model_name="ensemble_model",
                                  inputs=input_tensors,
                                  outputs=output)

    print(query_response.as_numpy("ensemble_output_string"))
    print(query_response.as_numpy("ensemble_output_UINT8_array"))
    print(query_response.as_numpy("ensemble_output_INT8_array"))
    print(query_response.as_numpy("ensemble_output_FP32_image"))
    print(query_response.as_numpy("ensemble_output_bool"))
```

看一下如何设置输入和输出。
```
# Input
input_tensors = [
    httpclient.InferInput("ensemble_input_string", text_obj.shape,np_to_triton_dtype(text_obj.dtype)),
    httpclient.InferInput("ensemble_input_UINT8_array", uint8_array.shape, datatype="UINT8"),
    httpclient.InferInput("ensemble_input_INT8_array", int8_array.shape, datatype="INT8"),
    httpclient.InferInput("ensemble_input_FP32_image", image.shape, datatype="FP32"),
    httpclient.InferInput("ensemble_input_bool", boolean.shape, datatype="BOOL")
]
input_tensors[0].set_data_from_numpy(text_obj)
input_tensors[1].set_data_from_numpy(uint8_array)
input_tensors[2].set_data_from_numpy(int8_array)
input_tensors[3].set_data_from_numpy(image)
input_tensors[4].set_data_from_numpy(boolean)

# Output
output = [
    httpclient.InferRequestedOutput("ensemble_output_string"),
    httpclient.InferRequestedOutput("ensemble_output_UINT8_array"),
    httpclient.InferRequestedOutput("ensemble_output_INT8_array"),
    httpclient.InferRequestedOutput("ensemble_output_FP32_image"),
    httpclient.InferRequestedOutput("ensemble_output_bool")
]

```
这里我们使用 `http` 客户端，指定输入输出的名字和预期的数据类型。注意这里用的是 ensemble 的输入输出，例如字符串输入用的是 `ensemble_input_string`。如果你想单独查询其中一个组成模型，把输入名、输出名和模型名改成对应模型即可。

```
# Creating a client for the server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Querying the Server
query_response = client.infer(model_name="ensemble_model",
                                inputs=input_tensors,
                                outputs=output)

print(query_response.as_numpy("ensemble_output_string"))
print(query_response.as_numpy("ensemble_output_UINT8_array"))
print(query_response.as_numpy("ensemble_output_INT8_array"))
print(query_response.as_numpy("ensemble_output_FP32_image"))
print(query_response.as_numpy("ensemble_output_bool"))
```

## 运行示例

要运行这个示例，需要打开两个终端。
```
# Server

cd /path/to/this/folder
# Replace yy.mm with year and month of release. Eg. 23.02
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
tritonserver --model-repository=/models
```
上面的命令会启动 Triton Inference Server。在第二个终端中，运行客户端脚本：
```
# Client

cd /path/to/this/folder
# Replace yy.mm with year and month of release. Eg. 23.02
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash
pip install image
python3 client.py
```

你的 ensemble 有带条件分支的流程吗？看看[这个示例](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_6-building_complex_pipelines)和 [Business Logic Scripting API 文档](https://github.com/triton-inference-server/python_backend#business-logic-scripting)！
