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


# 动态批处理与并发模型执行（Dynamic Batching & Concurrent Model Execution）

| 跳转到 | [第 1 部分：模型部署](../Part_1-model_deployment/) | [第 3 部分：优化 Triton 配置](../Part_3-optimizing_triton_configuration/) |
| ------------ | --------------- | --------------- |

本系列的第 1 部分介绍了如何搭建 Triton Inference Server。本部分讨论动态批处理（dynamic batching）和并发模型执行（concurrent model execution）这两个概念。它们是可以用来降低延迟、并通过更高的资源利用率提升吞吐的重要特性。

## 什么是动态批处理？

就 Triton Inference Server 而言，动态批处理指的是把一条或多条推理请求合并成单个批次（需要动态创建）以最大化吞吐的功能。

动态批处理可以按模型启用和配置，方法是在该模型的 `config.pbtxt` 中指定相应设置。在 `config.pbtxt` 中加入以下内容即可用默认配置启用动态批处理：

```
dynamic_batching { }
```

虽然 Triton 会在没有任何延迟的情况下对入站请求进行批处理，但用户也可以为调度器分配一个有限时间的延迟，让它收集更多推理请求供动态批处理器使用。

```
dynamic_batching {
    max_queue_delay_microseconds: 100
}
```

我们来讨论一个示例场景（参考下图）。假设有 5 条推理请求 `A`、`B`、`C`、`D` 和 `E`，它们的批大小分别是 `4`、`2`、`2`、`6` 和 `2`。每个批次被模型处理需要 `X ms`。模型支持的最大批大小是 `8`。`A` 和 `C` 在 `T = 0` 时到达，`B` 在 `T = X/3` 时到达，`D` 和 `E` 在 `T = 2*X/3` 时到达。

![Dynamic Batching Sample](./img/dynamic_batching.PNG)

在不使用动态批处理的情况下，所有请求串行处理，处理完所有请求需要 `5X ms`。这个过程相当浪费，因为每次批次处理本可以处理比串行执行时更多的批次。

而使用动态批处理可以把请求更高效地打包进 GPU 内存，处理时间显著缩短到 `3X ms`。同时它也降低了响应延迟，因为更少的处理周期内可以完成更多查询。如果考虑使用 `delay`，`A`、`B`、`C` 可以合批，`D`、`E` 也可以合批，从而获得更好的资源利用率。

**注意（Note）：** 以上是理想情况的极端版本。实际上，执行过程中的要素无法被完全并行化，批越大执行时间越长。

> 💡 **AI Infra 视角**：动态批处理是 GPU 利用率的第一杠杆。GPU 的算力高度依赖"批大小"——批太小，SM（流式多处理器）喂不满数据，算力就闲置了。在线推理场景请求到达是稀疏且不齐整的，动态批处理就是把这些零散请求攒成整齐的 batch。实践中要平衡 `max_queue_delay_microseconds`：延迟等太久会伤害 p99 延迟，等太短又凑不齐批。这就是典型的"延迟 vs 吞吐"权衡。

从上面的分析可以看出，在服务模型时使用动态批处理可以同时改善延迟和吞吐。这个批处理特性主要面向无状态模型（stateless models，即执行之间不维持状态的模型，如目标检测模型）。Triton 的[序列批处理器（sequence batcher）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md#sequence-batcher)可以用来管理有状态模型的多个推理请求。关于动态批处理的更多信息和配置，请参考 Triton Inference Server [文档](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md#dynamic-batcher)。

## 并发模型执行（Concurrent model execution）

Triton Inference Server 可以启动同一模型的多个实例，并行处理查询。Triton 可以在同一设备（GPU）上或同一节点的不同设备上按用户规格生成实例。这种可定制性在考虑由吞吐量不同的模型组成的集成（ensemble）时尤其有用。可以在单独的 GPU 上生成多个较重模型的副本，以支持更多并行处理。这是通过模型配置中的 `instance groups` 选项启用的。

```
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]
```

我们来沿用之前的例子，讨论加入多个模型并行执行的效果。在这个例子中，不再是单个模型处理 5 个查询，而是生成两个模型实例。![Multiple Model Instances](./img/multi_instance.PNG)

在"无动态批处理"的情况下，由于有两个模型实例，查询被平均分配。用户还可以添加[优先级（priorities）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#priority)来提升或降低某个实例组的优先级。

再考虑多个实例配合动态批处理的情况，会发生下面这些事情。由于有第二个实例可用，稍晚到达的查询 `B` 可以由第二个实例执行。分配了一定延迟后，实例 1 在 `T = X/2` 时被填满并启动，而由于查询 `D` 和 `E` 堆叠起来填满了最大批大小，第二个模型可以无延迟地开始推理。

> 💡 **AI Infra 视角**：instance group 是 Triton 里"用显存换并行度"的手段。单个模型实例常常喂不饱 GPU（尤其小模型），多实例让计算和显存更充分地利用。但注意：实例数开太多会让每个实例的批变小，可能适得其反。更关键的是它会按实例数翻倍占用显存——对 LLM 这种显存大户，多实例要谨慎，这也是为什么后面的第 3 部分要用 Model Analyzer 做系统性搜索而不是凭感觉调参。

从以上例子得出的关键结论是：Triton Inference Server 在如何构建更高效的批处理策略上提供了灵活性，从而带来更好的资源利用率、更低的延迟和更高的吞吐。

## 演示（Demonstration）

本节用本系列第 1 部分的示例演示动态批处理和并发模型执行的使用。

### 获取模型

我们使用第 1 部分用到的 `文本识别` 模型。需要对该模型做一些小改动，即让模型的第 0 维具有动态形状以支持批处理。第 1 步，下载文本识别模型的权重。以下步骤请使用 NGC PyTorch 容器作为环境。

```
docker run -it --gpus all -v ${PWD}:/scratch nvcr.io/nvidia/pytorch:<yy.mm>-py3
cd /scratch
wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth
```

使用 `utils` 文件夹中的模型定义文件将模型导出为 `.onnx`。该文件改编自 [Baek et. al. 2019](https://github.com/clovaai/deep-text-recognition-benchmark)。

```
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
torch.onnx.export(model, trace_input, "str.onnx", verbose=True, dynamic_axes={'input.1':[0],'308':[0]})
```

### 启动服务器

如 `第 1 部分` 所述，模型仓库是 Triton Inference Server 使用的基于文件系统的模型和配置集合（模型仓库的详细解释请参考 `第 1 部分`）。本示例中，模型仓库结构需要按如下方式搭建：

```
model_repository
|
|-- text_recognition
    |
    |-- config.pbtxt
    |-- 1
        |
        |-- model.onnx
```

这个仓库是之前示例的子集。与之前设置的关键区别在于模型配置中使用了 `instance_group`（实例组）和 `dynamic_batching`。新增内容如下：

```
instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
]
dynamic_batching { }
```

通过 `instance_group`，用户可以主要调整两件事。第一，该模型在每个 GPU 上部署的实例数量。上面的示例会在`每个 GPU` 上部署 `2` 个模型实例。第二，可以用 `gpus: [ <device number>, ... <device number> ]` 指定该组的 GPU 目标。

加上 `dynamic_batching {}` 就可以启用动态批处理。用户还可以在 dynamic batching 的配置体中添加 `preferred_batch_size` 和 `max_queue_delay_microseconds`，以便按自己的用例实现更高效的批处理。更多信息请查阅[模型配置](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration)文档。

模型仓库搭建好后，就可以启动 Triton Inference Server 了。

```
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash

tritonserver --model-repository=/models
```

### 测量性能

通过启用 `动态批处理` 和使用`多个模型实例`，我们对模型的 serving 能力做了一些改进，下一步是测量这些特性带来的影响。为此，Triton Inference Server 自带了 [Performance Analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md)，这是一个专门用来测量 Triton Inference Server 性能的工具。为方便使用，建议用户在本系列第 1 部分运行客户端代码的同一个容器里运行它。

```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash
```

建议在第三个终端里监控 GPU 利用率，看看部署是否吃满了 GPU 资源。

```
watch -n0.1 nvidia-smi
```

为了测量性能收益，我们在以下配置上运行性能分析器：

* **无动态批处理、单模型实例**：该配置作为基线测量。要按此配置搭建 Triton 服务器，不要在 `config.pbtxt` 中添加 `instance_group` 或 `dynamic_batching`，并确保 `docker run` 命令中包含 `--gpus=1` 来搭建服务器。

```
# perf_analyzer -m <model name> -b <batch size> --shape <input layer>:<input shape> --concurrency-range <lower number of request>:<higher number of request>:<step>

# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

# Summarized Inference Result
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 955.708 infer/sec, latency 4311 usec
Concurrency: 4, throughput: 977.314 infer/sec, latency 8497 usec
Concurrency: 6, throughput: 973.367 infer/sec, latency 12799 usec
Concurrency: 8, throughput: 974.623 infer/sec, latency 16977 usec
Concurrency: 10, throughput: 975.859 infer/sec, latency 21199 usec
Concurrency: 12, throughput: 976.191 infer/sec, latency 25519 usec
Concurrency: 14, throughput: 966.07 infer/sec, latency 29913 usec
Concurrency: 16, throughput: 975.048 infer/sec, latency 34035 usec

# Perf for 16 concurrent requests
Request concurrency: 16
  Client:
    Request count: 8777
    Throughput: 975.048 infer/sec
    p50 latency: 32566 usec
    p90 latency: 33897 usec
    p95 latency: 34035 usec
    p99 latency: 34241 usec
    Avg HTTP time: 32805 usec (send/recv 43 usec + response wait 32762 usec)
  Server:
    Inference count: 143606
    Execution count: 71803
    Successful request count: 71803
    Avg request latency: 17937 usec (overhead 14 usec + queue 15854 usec + compute input 20 usec + compute infer 2040 usec + compute output 7 usec)
```

* **仅动态批处理**：要按此配置搭建 Triton 服务器，在 `config.pbtxt` 中添加 `dynamic_batching`。

```
# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

# Inference Result
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 998.141 infer/sec, latency 4140 usec
Concurrency: 4, throughput: 1765.66 infer/sec, latency 4750 usec
Concurrency: 6, throughput: 2518.48 infer/sec, latency 5148 usec
Concurrency: 8, throughput: 3095.85 infer/sec, latency 5565 usec
Concurrency: 10, throughput: 3182.83 infer/sec, latency 7632 usec
Concurrency: 12, throughput: 3181.3 infer/sec, latency 7956 usec
Concurrency: 14, throughput: 3184.54 infer/sec, latency 10357 usec
Concurrency: 16, throughput: 3187.76 infer/sec, latency 10567 usec

# Perf for 16 concurrent requests
Request concurrency: 16
  Client:
    Request count: 28696
    Throughput: 3187.76 infer/sec
    p50 latency: 10030 usec
    p90 latency: 10418 usec
    p95 latency: 10567 usec
    p99 latency: 10713 usec
    Avg HTTP time: 10030 usec (send/recv 54 usec + response wait 9976 usec)
  Server:
    Inference count: 393140
    Execution count: 64217
    Successful request count: 196570
    Avg request latency: 6231 usec (overhead 31 usec + queue 3758 usec + compute input 35 usec + compute infer 2396 usec + compute output 11 usec)
```

由于每条请求的批大小是 `2`，而模型的最大批大小是 `8`，对这些请求做动态批处理带来了显著的吞吐提升。另一个结果是延迟下降。这种下降主要归因于队列等待时间减少。请求被合批后，多个请求可以并行处理。

* **动态批处理 + 多个模型实例**：要按此配置搭建 Triton 服务器，在 `config.pbtxt` 中添加 `instance_group`，确保 `docker run` 命令中包含 `--gpus=1`。并按照上一节说明在模型配置中加入 `dynamic_batching`。值得注意的一点是，仅用单模型实例加动态批处理时，GPU 利用率峰值已经冲到了 74%（本例中是 A100）。再加一个实例肯定会改善性能，但在这个场景下无法实现线性性能扩展。

```
# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95

# Inference Result
Inferences/Second vs. Client p95 Batch Latency
Concurrency: 2, throughput: 1446.26 infer/sec, latency 3108 usec
Concurrency: 4, throughput: 1926.1 infer/sec, latency 5491 usec
Concurrency: 6, throughput: 2695.12 infer/sec, latency 5710 usec
Concurrency: 8, throughput: 3224.69 infer/sec, latency 6268 usec
Concurrency: 10, throughput: 3380.49 infer/sec, latency 6932 usec
Concurrency: 12, throughput: 3982.13 infer/sec, latency 7233 usec
Concurrency: 14, throughput: 4027.74 infer/sec, latency 7879 usec
Concurrency: 16, throughput: 4134.09 infer/sec, latency 8244 usec

# Perf for 16 concurrent requests
Request concurrency: 16
  Client:
    Request count: 37218
    Throughput: 4134.09 infer/sec
    p50 latency: 7742 usec
    p90 latency: 8022 usec
    p95 latency: 8244 usec
    p99 latency: 8563 usec
    Avg HTTP time: 7734 usec (send/recv 54 usec + response wait 7680 usec)
  Server:
    Inference count: 490626
    Execution count: 101509
    Successful request count: 245313
    Avg request latency: 5287 usec (overhead 29 usec + queue 1878 usec + compute input 36 usec + compute infer 3332 usec + compute output 11 usec)
```

这是一个绝佳的例子，说明"简单地把所有特性打开"并不是放之四海而皆准的方案。需要指出的是，本次实验把模型的最大批大小限制为 `8`，并且是单 GPU 环境。每个生产环境都不一样。模型、硬件、业务级 SLA、成本，都是在选择部署配置时需要纳入考量的变量。对每个部署都做网格搜索（grid search）并不是可行的策略。为了解决这个难题，Triton 用户可以使用本教程第 3 部分要讲的 Model Analyzer！也可以看看[文档的这一节](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md#optimization)，那里有动态批处理和多个模型实例的另一个示例。

> 💡 **AI Infra 视角**：注意这三组数字的对比：基线 975 infer/sec → 动态批处理 3187 infer/sec（约 3.3 倍）→ 加多实例 4134 infer/sec（再提升约 30%）。两个启发值得记住：第一，动态批处理通常是投入产出比最高的优化；第二，多实例带来的收益随 GPU 利用率的饱和而递减——配置收益不是线性的，这正是需要自动化工具搜索配置空间的原因。

# 接下来是什么？

本教程我们讲了两个可以用来提升资源利用率的核心概念：`动态批处理` 和 `并发模型执行`。这是 6 部分教程系列的第 2 部分，该系列讨论的是将深度学习模型部署到生产环境所面临的挑战。你可能已经意识到，本教程讨论的特性可以组合出很多可能性，尤其是在多 GPU 节点上。第 3 部分将介绍 `Model Analyzer`，一个帮助找到最佳部署配置的工具。
