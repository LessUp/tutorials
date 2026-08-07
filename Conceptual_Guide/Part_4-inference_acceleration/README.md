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


# 深度学习模型的推理加速（Accelerating Inference for Deep Learning Models）

| 跳转到 | [第 3 部分：优化 Triton 配置](../Part_3-optimizing_triton_configuration/)  | [第 5 部分：构建模型集成](../Part_5-Model_Ensembles/) |
| ------------ | --------------- | --------------- |

模型加速是一个复杂而微妙的话题。图优化（graph optimization）、剪枝（pruning）、知识蒸馏（knowledge distillation）、量化（quantization）等技术的可行性高度依赖于模型的结构。这些主题每一个本身都是庞大的研究领域，而构建定制工具需要巨大的工程投入。

与其对生态做穷尽式罗列，为简洁和客观起见，本部分将聚焦于使用 Triton Inference Server 部署模型时推荐使用的工具和特性。

![Triton Flow](./img/query_flow.PNG)

Triton Inference Server 有一个叫做 "Triton Backend"（简称 ["Backend"](https://github.com/triton-inference-server/backend)）的概念。Backend 本质上是执行模型的实现。一个 backend 可以是热门深度学习框架（如 PyTorch、TensorFlow、TensorRT 或 ONNX Runtime）的封装，用户也可以构建针对自己的模型和用例定制化的 backend。每个 backend 都有各自的加速选项。

Triton 模型的性能调优在[这里](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/performance_tuning.md)有广泛讨论，但本文档会在下面展开更多细节。

加速建议取决于两个主要因素：
* **硬件类型**：Triton 用户可以选择在 GPU 或 CPU 上运行模型。凭借其提供的并行能力，GPU 提供了许多性能加速的途径。使用 PyTorch、TensorFlow、ONNX Runtime 和 TensorRT 的模型可以利用这些优势。对于 CPU，Triton 用户可以利用 OpenVINO backend 进行加速。
* **模型类型**：用户通常使用三类模型中的一类或多类：`浅层模型（Shallow models）`，如随机森林；`神经网络（Neural Networks）`，如 BERT 或 CNN；最后是`大型 Transformer 模型（Large Transformer Models）`，它们通常太大，无法装进单块 GPU 的显存。每类模型利用不同的优化手段来加速性能。

![Decision Tree](./img/selecting_accelerator.PNG)

有了这些大类别的考虑，让我们深入具体的场景和决策过程，为用户用例挑选最合适的 Triton Backend，并简要讨论可能的优化。

## GPU 加速

如前所述，深度学习模型的加速可以通过多种方式实现。融合层（fusing layers）等图级别的优化可以减少执行时需要启动的 GPU kernel 数量。融合层让模型执行更省内存，并提高了操作密度。融合后，kernel 自动调优器（auto tuner）可以挑选正确的 kernel 组合来最大化 GPU 资源利用率。同样，配合量化等技术使用更低精度（FP16、INT8 等）可以大幅降低显存需求并提高吞吐。

性能优化策略的确切性质因 GPU 的硬件设计而异。这些都是我们用 [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) 为深度学习从业者解决的一小部分挑战——TensorRT 是一个专注于深度学习推理优化的 SDK。

TensorRT 可以与 PyTorch、TensorFlow、MxNET、ONNX Runtime 等热门深度学习框架协同工作，同时它也提供框架级集成：PyTorch（[Torch-TensorRT](https://github.com/pytorch/TensorRT)）和 TensorFlow（[TensorFlow-TensorRT](https://github.com/tensorflow/tensorrt)），为各自的开发者提供灵活性和回退机制。

> 💡 **AI Infra 视角**：模型加速的本质是在"可接受的精度损失"内换取"更低的延迟和更高的吞吐"。GPU 推理的优化手段可以粗略分为两层：图优化（把算子融合成更少、更大的 kernel）和精度优化（FP16/INT8 量化）。在推理平台上，加速通常由 TensorRT 这类引擎完成，但要注意：引擎文件与 GPU 架构（compute capability）绑定，换卡型号必须重新构建——这是生产环境常踩的坑。

### 直接使用 TensorRT

用户把模型转换为 TensorRT 有三条路径：C++ API、Python API，以及 [trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)/[polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)（TensorRT 的命令行工具）。[参考这份指南看完整示例](https://github.com/NVIDIA/TensorRT/tree/main/quickstart/deploy_to_triton)。

话虽如此，主要有两个步骤。第一步，把模型转换为 TensorRT 引擎（Engine）。建议使用 [TensorRT 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)来运行命令。

```
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --explicitBatch
```

转换完成后，按第 1 部分所述把 `model.plan` 放到模型仓库中，并在 `config.pbtxt` 中使用 `tensorrt` 作为 `backend`。

除了转换到 TensorRT，用户还可以利用一些 [cuda 特有的优化](https://github.com/triton-inference-server/common/blob/d4017443199e4f19462360789f5c80b0eb1e4738/protobuf/model_config.proto#L823)。

如果用户的模型中有一些算子不受 TensorRT 支持，有三种可能的方案：
* **使用框架集成之一**：TensorRT 有两个框架集成：Torch-TensorRT（PyTorch）和 TensorFlow-TensorRT（TensorFlow）。这些集成内置了回退机制，在 TensorRT 不直接支持该图时使用框架 backend。

* **在 ONNX Runtime 中使用 TensorRT**：Triton 用户也可以利用 ONNX Runtime 的这个回退机制（下一节详述）。

* **构建插件（plugin）**：TensorRT 允许构建插件并实现自定义算子。用户可以编写自己的 [TensorRT 插件](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)来实现不受支持的算子（建议专家用户使用）。非常鼓励把这类算子反馈给 NVIDIA，让它们在 TensorRT 中原生支持。

### 使用 TensorRT 与 PyTorch/TensorFlow 的集成

对于 **PyTorch**，Torch-TensorRT 是一个提前编译（Ahead of Time，AOT）编译器，把 TorchScript/Torch FX 转换为面向 TensorRT 引擎的模块。编译完成后，用户可以用与使用 TorchScript 模型相同的方式使用优化后的模型。观看[这个入门视频](https://www.youtube.com/watch?v=TU5BMU6iYZ0)了解更多。关于用 Torch TensorRT 编译 PyTorch 模型并部署到 Triton 的完整示例，[参考这份指南](https://pytorch.org/TensorRT/tutorials/serving_torch_tensorrt_with_triton.html)。

**TensorFlow** 用户可以使用 TensorFlow TensorRT：它把图切分为 TensorRT 支持和不支持的子图。受支持的子图随后被替换为 TensorRT 优化节点，生成一个同时包含 TensorFlow 和 TensorRT 组件的图。[参考这个教程](https://github.com/tensorflow/tensorrt/tree/master/tftrt/triton)，它详细说明了用 TensorFlow-TensorRT 加速模型并部署到 Triton Inference Server 的确切步骤。

![Flow](./img/fw-trt-workflow.PNG)

### 使用 TensorRT 与 ONNX Runtime 的集成

加速 ONNX Runtime 有三种方案：GPU 上用 `TensorRT` 和 `CUDA` 执行提供器（Execution Provider，EP），CPU 上用 `OpenVINO`（后面章节讨论）。

一般来说，TensorRT 提供的优化优于 CUDA 执行提供器，不过这取决于模型的确切结构，更准确地说，取决于被加速网络中使用的算子。如果所有算子都受支持，转换为 TensorRT 会带来更好的性能。当选择 `TensorRT` 作为加速器时，所有受支持的子图由 TensorRT 加速，图的其余部分在 CUDA 执行提供器上运行。用户可以在配置文件中加入以下内容来实现这一点。

**TensorRT 加速**
```
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
  }
}
```

也就是说，用户也可以选择不带 TensorRT 优化运行模型，此时 CUDA EP 是默认执行提供器。更多细节见[这里](https://github.com/triton-inference-server/onnxruntime_backend#onnx-runtime-with-tensorrt-optimization)。本系列第 1-3 部分使用的 `文本识别` 模型的示例配置文件见 `onnx_tensorrt_config.pbtxt`。

ONNX Runtime 还有一些其他特定优化。更多信息请参考我们的 [ONNX backend 文档的这一节](https://github.com/triton-inference-server/onnxruntime_backend#other-optimization-options-with-onnx-runtime)。

## CPU 加速

Triton Inference Server 也支持用 [OpenVINO](https://docs.openvino.ai/latest/index.html) 加速纯 CPU 模型。在配置文件中加入以下内容即可启用 CPU 加速。

```
optimization {
  execution_accelerators {
    cpu_execution_accelerator : [{
      name : "openvino"
    }]
  }
}
```

OpenVINO 提供软件层面的优化，同时 CPU 硬件本身也很重要。CPU 由多核、内存资源和互连组成。多路 CPU 的情况下，这些资源可以通过 NUMA（非统一内存访问，Non Uniform Memory Access）共享。更多内容请参考 [Triton 文档的这一节](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#numa-optimization)。

## 加速浅层模型（Shallow models）

梯度提升决策树（Gradient Boosted Decision Trees）这类浅层模型经常出现在各种流水线中。这些模型通常用 [XGBoost](https://xgboost.readthedocs.io/en/stable/)、[LightGBM](https://lightgbm.readthedocs.io/en/stable/)、[Scikit-learn](https://scikit-learn.org/stable/)、[cuML](https://github.com/rapidsai/cuml) 等库构建。这些模型可以通过 Forest Inference Library（FIL）backend 部署到 Triton Inference Server 上。[这些示例](https://github.com/triton-inference-server/fil_backend/tree/main/notebooks)提供了更多信息。

> 💡 **AI Infra 视角**：别以为推理平台只服务深度学习模型——工业界大量场景（风控、广告 CTR 预估）的主力是树模型。FIL backend 的价值在于把 XGBoost/LightGBM 模型也用 Triton 的统一接口管起来，让团队在同一个 serving 层上管理所有模型，运维和监控口径一致。

## 加速大型 Transformer 模型

光谱的另一端，深度学习从业者被数十亿参数的大型 Transformer 模型吸引。这种规模的模型常常需要不同类型的优化，或需要跨 GPU 并行化。跨 GPU 并行化（因为模型可能无法装进 1 块 GPU）可以通过张量并行（Tensor parallelism）或流水线并行（Pipeline parallelism）实现。为解决这个问题，用户可以使用 [Faster Transformer 库](https://github.com/NVIDIA/FasterTransformer/)和 Triton 的 [Faster Transformer Backend](https://github.com/triton-inference-server/fastertransformer_backend)。[看看这篇博客](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/)了解更多！

> 💡 **AI Infra 视角**：LLM 时代，这一步的内容已经演进出专门方案：TensorRT-LLM、vLLM 等引擎把张量并行、KV cache、continuous batching 集成进来。对部署工程师来说，理解"张量并行把单个权重矩阵切到多卡、流水线并行把层切到多卡"的差异很重要：前者强依赖卡间高速互联（NVLink），后者对互联要求低一些。今天部署 7B 以上模型时，这些仍是显存放不下的基本应对手段。

## 动手示例（Working Example）

开始前，请为本系列第 1-3 部分使用的文本识别模型搭建好模型仓库。然后，进入模型仓库目录，启动两个容器：

```
# Server Container
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:26.07-py3 bash

# Client Container (on a different terminal)
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:26.07-py3-sdk bash
```

由于这是一个我们转换为 ONNX 的模型，而 TensorRT 加速示例在讲解中已多次链接，我们将探索 ONNX 这条路。ONNX backend 有三种情况要考虑：
* GPU 上用 CUDA 执行提供器加速的 ONNX RT：`ORT_cuda_ep_config.pbtxt`
* GPU 上用 TRT 加速的 ONNX RT：`ORT_TRT_config.pbtxt`
* CPU 上用 OpenVINO 加速的 ONNX RT：`ORT_openvino_config.pbtxt`

使用 ONNX RT 时，无论执行提供器是哪一种，都有一些需要考虑的[通用优化](https://github.com/triton-inference-server/onnxruntime_backend#other-optimization-options-with-onnx-runtime)。它们可以是图级别的优化，或者是选择用于并行化执行的线程数量和行为，或者是一些内存使用优化。每个选项的使用高度依赖于被部署的模型。

带着这些背景，我们用合适的配置文件启动 Triton Inference Server。

```
tritonserver --model-repository=/models
```

**注意：这些基准测试只是为了展示性能提升的大致曲线。这不是 Triton 能获得的最高吞吐，因为资源利用特性（如动态批处理）没有启用。模型优化完成后，请参考 Model Analyzer 教程获取最佳部署配置。**

**注意**：这些设置是为了最大化吞吐。管理延迟需求的内容请参考 Model Analyzer 教程。

作为参考，基线性能如下：

```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 4191.7 infer/sec, latency 7633 usec
```

### GPU 上用 CUDA 执行提供器运行 ONNX RT

对于这个模型，启用了对最佳卷积算法的穷举搜索。[了解其他选项](https://github.com/triton-inference-server/onnxruntime_backend#onnx-runtime-with-cuda-execution-provider-optimization)。

```
## Additions to Config
parameters { key: "cudnn_conv_algo_search" value: { string_value: "0" } }
parameters { key: "gpu_mem_limit" value: { string_value: "4294967200" } }

## Perf Analyzer Query
perf_analyzer -m text_recognition -b 16 --shape input.1:1,32,100 --concurrency-range 64
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 4257.9 infer/sec, latency 7672 usec
```

### GPU 上用 TRT 加速运行 ONNX RT

指定使用 TensorRT 执行提供器时，TensorRT 不支持的算子由 CUDA 执行提供器回退处理。如果所有算子都受支持，建议原生使用 TensorRT，因为性能提升和优化选项要好得多。本例中，TensorRT 加速器使用了更低的 `FP16` 精度。

```
## Additions to Config
optimization {
  graph : {
    level : 1
  }
 execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt",
      parameters { key: "precision_mode" value: "FP16" },
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
  }
}

## Perf Analyzer Query
perf_analyzer -m text_recognition -b 16 --shape input.1:1,32,100 --concurrency-range 2
...
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 11820.2 infer/sec, latency 2706 usec
```

### CPU 上用 OpenVINO 加速运行 ONNX RT

Triton 用户也可以使用 OpenVINO 做 CPU 部署。可以通过以下配置启用：

```
optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "openvino"
  } ]
}}
```

由于大多数情况下 1 个 CPU 和 1 个 GPU 并不是对等的比较，我们建议在用户自己的 CPU 硬件上做基准测试。[了解更多](https://github.com/triton-inference-server/onnxruntime_backend#onnx-runtime-with-openvino-optimization)

每个 backend 还有很多可以根据特定模型需求启用的特性。完整的特性和优化列表请参考[这个 protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)。

## Model Navigator

上面几节描述了模型转换和使用不同加速器的方法，为考虑优化时该走哪条"路"建立了一个"总体指南"。这些是相当耗时的手动探索。为了检查转换覆盖率并探索一部分可能的优化，用户可以使用 [Model Navigator 工具](https://github.com/triton-inference-server/model_navigator)。

# 接下来是什么？

本教程我们讲了在使用 Triton Inference Server 时可用于加速模型的大量优化选项。这是 6 部分教程系列的第 4 部分，该系列讨论的是将深度学习模型部署到生产环境所面临的挑战。第 5 部分讲的是`构建模型集成`。第 3 部分和第 4 部分分别聚焦两个不同的方面：资源利用率和框架级模型加速。把这两种技术结合起来使用会带来最佳性能。由于具体选择高度依赖工作负载、模型、SLA 和硬件资源，这个过程因人而异。我们强烈鼓励用户尝试所有这些特性，找到最适合自己用例的部署配置。
