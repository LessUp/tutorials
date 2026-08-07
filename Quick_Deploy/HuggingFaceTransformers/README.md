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

# 在 Triton 中部署 Hugging Face Transformer 模型

本教程演示如何借助 Triton 的 [Python 后端](https://github.com/triton-inference-server/python_backend)，在 Triton Inference Server 上部署任意的 Hugging Face transformer 模型。本例将部署以下 transformer 模型：
- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b)

选择这些模型是因为它们人气高、生成质量稳定。不过，只要基础设施足够，本教程的方法同样适用于任意 transformer 模型。

*注意*：本教程仅作为参考示例，可能未针对性能做过调优。

*注意*：下面的步骤中不会专门提到 Llama 2 模型，但只要把 `tiiuae/falcon-7b` 换成 `meta-llama/Llama-2-7b-hf`、把 `falcon7b` 文件夹换成 `llama7b` 文件夹，即可运行。

## 第一步：创建模型仓库

第一步是创建模型仓库，让 Triton Inference Server 加载其中的模型并进行推理。为此，先创建一个名为 `model_repository` 的目录，把 `falcon7b` 模型文件夹复制进去：

```
mkdir -p model_repository
cp -r falcon7b/ model_repository/
```

我们复制的 `falcon7b/` 文件夹按照 Triton 约定的方式组织，包含在 Triton 中服务模型所需的两份重要文件：
- **config.pbtxt** - 声明要使用的后端、模型的输入输出信息，以及执行时的自定义参数。关于 Triton 支持的模型配置属性的完整说明，请参见[这里](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)。
- **model.py** - 实现 Triton 在初始化（initialization）、执行（execution）和收尾（finalization）三个阶段应如何处理该模型。关于 Python 后端的更多用法，请参见[这里](https://github.com/triton-inference-server/python_backend#usage)。

> 💡 **AI Infra 视角**：Python 后端允许你用普通 Python 代码（这里是 Hugging Face transformers 的 pipeline）实现完整的推理逻辑，几乎零门槛即可上线任意模型。代价是性能——逐请求的 Python 解释开销和 GIL 限制使其吞吐远低于 TensorRT、ONNX 等专用后端。生产中的常见取舍是：先用 Python 后端快速验证业务，待流量增长后再把核心网络迁移到专用后端，保留 Python 只做前后处理。

## 第二步：构建 Triton 容器镜像

第二步是构建一个包含部署 Hugging Face transformer 模型所需全部依赖的镜像。用仓库提供的 Dockerfile 构建即可：

```
docker build -t triton_transformer_server .
```

## 第三步：启动 Triton Inference Server

`triton_transformer_server` 镜像构建完成后，可以用下面的命令在容器中启动 Triton Inference Server：
```bash
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository
```

**注意**：对于 `Llama2` 这类私有模型，你需要先[申请模型访问权限](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)，再把[访问令牌](https://huggingface.co/settings/tokens)通过 docker 命令的 `-e PRIVATE_REPO_TOKEN=<hf_your_huggingface_access_token>` 传入容器。
```bash
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -e PRIVATE_REPO_TOKEN=<hf_your_huggingface_access_token> -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server tritonserver --model-repository=model_repository
```

> 💡 **AI Infra 视角**：服务器启动时会把整个模型加载进显存——以 falcon-7b 为例，7B 参数用 FP16 存储约需 14GB 显存，因此至少要 A100 40GB 或两张 24GB 显卡才能容下模型权重加上推理时的 KV 缓存。启动命令中的 `--shm-size=1G` 与 `--ulimit memlock=-1` 也不是可有可无：Python 后端通过共享内存（shared memory）在进程间传递张量数据，memlock 限制解除后则允许模型权重锁定在内存中，避免被换页拖慢推理。

看到控制台出现以下输出时，说明服务器已成功启动：

```
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

## 第四步：查询服务器

现在可以用 curl 查询服务器，指定服务器地址和输入内容：

```bash
curl -X POST localhost:8000/v2/models/falcon7b/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["I am going"]}]}'
```
在我们的测试中，服务器返回了如下结果（为便于阅读已格式化）：
```json
{
  "model_name": "falcon7b",
  "model_version": "1",
  "outputs": [
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [
        1
      ],
      "data": [
        "I am going to be in the market for a new laptop soon. I"
      ]
    }
  ]
}
```

## 第五步：在 Triton 中托管多个模型

到目前为止，本教程只加载了一个模型。其实 Triton 可以同时托管多个模型。要加载更多模型，先按 `Ctrl+C` 退出 docker 容器并等待其退出。

接下来，把剩下的模型复制进模型仓库：
```
cp -r persimmon8b/ model_repository/
```
*注意*：这两个模型加起来的体积很大。如果你的硬件无法同时支撑两个模型，可以考虑加载更小的模型，例如 [opt-125m](https://huggingface.co/facebook/opt-125m)——用提供的模板为它创建一个文件夹并复制到 `model_repository` 即可。

> 💡 **AI Infra 视角**：多模型共存是 Triton 的核心卖点之一：一套服务同时承载多个模型，通过调度器共享 GPU 资源，避免为每个模型单独起服务造成的显存碎片和 GPU 空转。但"能共存"不等于"应该共存"——每个模型都会占用显存，需要根据模型大小和流量预估做好容量规划，必要时用显存监控（如 `nvidia-smi` 或 Triton 8002 端口的指标）验证预算是否够用。

再次用上面的 `docker run` 命令启动服务器，并等待服务器成功启动的确认信息。

查询服务器时，注意为每个模型修改请求地址：
```bash
curl -X POST localhost:8000/v2/models/falcon7b/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["How can you be"]}]}'
curl -X POST localhost:8000/v2/models/persimmon8b/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["Where is the nearest"]}]}'
```
在我们的测试中，这两个查询返回了以下结果：
```bash
# falcon7b
"How can you be sure that you are getting the best deal on your car"

# persimmon8b
"Where is the nearest starbucks?"
```
从 23.10 版本开始，用户可以通过 Triton 的 generate 端点，以更简化的方式与 Triton 托管的大语言模型（LLM）交互：

```bash
curl -X POST localhost:8000/v2/models/falcon7b/generate -d '{"text_input":"How can you be"}'
```
## 对最新模型的即时支持（'Day Zero' Support）

最新的 transformer 模型可能无法立即获得官方 `transformers` 包的支持。这种情况下，你可以通过从源码构建 `transformers`，让这些"前沿"模型也能在 Triton 中加载。做法是把提供的 Dockerfile 中的 transformers 安装指令替换为：
```docker
RUN pip install git+https://github.com/huggingface/transformers.git
```
用这种方法，你应该可以在 Triton 中服务 Hugging Face 支持的任何 transformer 模型。


## 后续步骤
下面的章节是对基础教程的扩展，为后续试验提供指导。

### 加载缓存模型
前面的步骤中，我们在启动 Triton 服务器时才从 Hugging Face 下载 falcon-7b 模型。后续运行可以通过加载缓存模型，省去这个耗时的下载过程。默认情况下，提供的 `model.py` 会通过设置 `TRANSFORMERS_CACHE` 环境变量，把 falcon 和 persimmon 模型缓存在 `model_repository` 文件夹中各自的目录里。要为任意模型设置该环境变量，请在 `model.py` 中 **导入 'transformers' 模块之前** 加入下面几行，并把 `{MODEL}` 替换为目标模型。

```python
import os
os.environ['TRANSFORMERS_CACHE'] = '/opt/tritonserver/model_repository/{MODEL}/hf_cache'
```

另外，如果你的系统已经缓存了要部署到 Triton 的 Hugging Face 模型，可以在之前的 `docker run` 命令中加上下面的挂载选项，把它挂载进 Triton 容器（把 `${HOME}` 替换为你用户的主目录路径）：

```bash
# 挂载某个特定缓存模型的选项（这里以 falcon-7b 为例）
-v ${HOME}/.cache/huggingface/hub/models--tiiuae--falcon-7b:/root/.cache/huggingface/hub/models--tiiuae--falcon-7b

# 挂载宿主机上所有缓存模型的选项
-v ${HOME}/.cache/huggingface:/root/.cache/huggingface
```

### Triton 工具生态
在 Triton 中部署模型，还能获得一套功能完整的部署分析工具，帮助你更好地了解和调优系统。Triton 目前提供两种部署分析工具：
- [Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html)：推理性能优化器。
- [Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html)：GPU 显存和计算利用率优化器。

#### Performance Analyzer
使用性能分析器前，请先从 `model_repository` 中移除 persimmon8b 模型，再用上面的 `docker run` 命令重启 Triton 服务器。

Triton 成功启动后，在另一个窗口中启动 Triton SDK 容器：

```bash
docker run -it --net=host nvcr.io/nvidia/tritonserver:26.07-py3-sdk bash
```
这个容器预装了 Triton 的全部部署分析工具，因此只需输入下面的命令，就能得到模型推理性能的反馈：

```bash
perf_analyzer -m falcon7b --collect-metrics
```

这个命令运行得很快，会对我们的 falcon7b 模型进行性能剖析。分析器运行时会输出延迟百分位、推理各阶段延迟、成功请求数等有用指标。部分输出数据如下：

```bash
#Avg request latency
46307 usec (overhead 25 usec + queue 25 usec + compute input 26 usec + compute infer 46161 usec + compute output 68 usec)

#Avg GPU Utilization
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 66.0556%

#Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 21.3841 infer/sec, latency 46783 usec
```

> 💡 **AI Infra 视角**：注意上面的耗时分解——`queue` 25 usec 表示请求在调度队列里的等待时间，`compute infer` 46161 usec 才是真正的 GPU 计算时间：单请求串行执行时 GPU 大部分时间在等待，利用率只有 66%。提升 GPU 利用率的标准手段就是批处理（batching）：把多个请求合并成一批一次计算，分摊启动开销，但这会牺牲单请求延迟，吞吐与延迟之间的平衡正是推理服务调优的核心课题。

这些指标说明我们还没有充分利用硬件，吞吐也很低。把逐个推理改为批量处理，可以立刻改善结果。falcon 模型的 `model.py` 已支持处理批量请求。在 Triton 中启用批处理很简单，只需在 falcon 的 `config.pbtxt` 中添加以下内容：

```
dynamic_batching { }
max_batch_size: 8
```

`max_batch_size` 对应的整数可以随意选择，本示例中我们选了 8。现在用不断增大的并发度重新运行 perf_analyzer，看看它如何影响 GPU 利用率和吞吐：

```bash
perf_analyzer -m falcon7b --collect-metrics --concurrency-range=2:16:2
```
运行几分钟后，性能分析器应返回类似下面的结果（取决于硬件）：
```bash
# Concurrency = 4
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 74.1111%
Throughput: 31.8264 infer/sec, latency 125174 usec

# Concurrency = 8
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 81.7895%
Throughput: 46.2105 infer/sec, latency 172920 usec

# Concurrency = 16
GPU-57c7b00e-ca04-3876-91e2-c1eae40a0733 : 90.5556%
Throughput: 53.6549 infer/sec, latency 299178 usec
```

借助性能分析器，我们快速剖析了不同的模型配置，获得了更好的吞吐和硬件利用率。在这个例子中，我们用了不到 5 分钟就找到了一份配置：吞吐提升近两倍，GPU 利用率提高约 24%。

这只是性能分析器的一个简单用例。关于性能分析器参数的更完整说明和更多用例，请参见[这份](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2310/user-guide/docs/user_guide/perf_analyzer.html)指南。

关于 Triton 动态批处理（dynamic batching）的更多信息，请参见[这份](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher)指南。

#### Model Analyzer

在性能分析器一节中，我们靠直觉修改了一小部分变量并对比性能差异来提升吞吐。但面对广阔的搜索空间，我们只试了几个变量。

要更系统地扫描参数空间，可以用 Triton 的 model analyzer——它不仅会扫描大量配置参数组合，还会生成可视化报告，供执行后分析。

使用 model analyzer 前，请先按 `Ctrl+C` 终止 Triton 服务器，再用下面的命令重新启动（确保 falcon 模型的 config.pbtxt 中已加入上面的 dynamic_batching 参数）：
```bash
docker run --gpus all -it --rm --net=host --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/model_repository:/opt/tritonserver/model_repository triton_transformer_server
```

接下来，为了从 model analyzer 获得最准确的 GPU 指标，我们将在本地服务器容器中安装并启动它。首先安装 model analyzer：
```bash
pip3 install triton-model-analyzer
```

安装成功后，输入下面的命令（如有必要，可把实例数调低以适配你的 GPU）：
```bash
model-analyzer profile -m /opt/tritonserver/model_repository/ --profile-models falcon7b --run-config-search-max-instance-count=3 --run-config-search-min-model-batch-size=8
```

这个工具的执行时间会比性能分析器示例长得多（约 40 分钟）。如果执行时间太长，也可以加上 `--run-config-search-mode quick` 选项运行。在我们的实验中，快速搜索模式产出的结果更少，但耗时减半。无论如何，model analyzer 完成后会以多种格式提供吞吐、延迟和硬件利用率的完整汇总。下面摘录了我们的运行结果，按性能排序：

| Model Config Name | Max Batch Size | Dynamic Batching | Total Instance Count | p99 Latency (ms) | Throughput (infer/sec) | Max GPU Memory Usage (MB) | Average GPU Utilization (%) |
| :---: | :----: | :---: | :----: | :---: | :----:   | :---: | :---: |
| falcon7b_config_7 | 16 | Enabled | 3:GPU | 1412.581 | 71.944 | 46226 | 100.0 |
| falcon7b_config_8 | 32 | Enabled | 3:GPU | 2836.225 | 63.9652 | 46268 | 100.0 |
| falcon7b_config_4 | 16 | Enabled | 2:GPU | 7601.437 | 63.9454 | 31331 | 100.0 |
| falcon7b_config_default | 8 | Enabled | 1:GPU | 4151.873 | 63.9384 | 16449 | 89.3 |

> 💡 **AI Infra 视角**：注意对比 `falcon7b_config_7`（3 个实例、p99 延迟 1412ms、吞吐 71.9/s）与 `falcon7b_config_4`（2 个实例、p99 延迟 7601ms、吞吐 63.9/s）——多实例并不必然带来低延迟，因为 GPU 算力被更多实例切分后单请求排队更久。模型实例数（instance count）、批大小与延迟之间没有简单的单调关系，这正是 Model Analyzer 存在的意义：它自动搜索配置空间，找到符合你 SLO 的目标函数最优解，省去手工试错。

我们可以通过查看详细报告，更细粒度地审视其中任一配置的性能。这类报告聚焦于单个配置在吞吐和硬件利用率下的延迟与并发指标。下面摘录了我们测试中表现最优的配置（为简洁起见已删减）：

| Request Concurrency | p99 Latency (ms) | Client Response Wait (ms) | Server Queue (ms) | Server Compute Input (ms) | Server Compute Infer (ms) | Throughput (infer/sec) | Max GPU Memory Usage (MB) | Average GPU Utilization (%) |
| :---: | :----: | :---: | :----: | :---: | :----:   | :---: | :---: | :---: |
| 512	| 8689.491 | 8190.506 | 7397.975 | 0.166 | 778.565 | 63.954 | 46230.667264 | 100.0 |
| | | | | ... | | | | |
| 128 | 2289.118 | 2049.37 | 1277.34 | 0.159 | 770.771 | 61.2953 | 46230.667264 | 100.0 |
| 64 | 1412.581 | 896.924 | 227.108 | 0.157 | 667.757 | 71.944 | 46226.47296 | 100.0 |
| 32 | 781.362 | 546.35 | 86.078 | 0.103 | 459.257 | 57.7877 | 46226.47296 | 100.0 |
| | | | | ... | | | | |
| 1 | 67.12 | 49.707 | 0.049 | 0.024 | 49.121 | 20.0993 | 46207.598592 | 54.9 |

同样，这只是 model analyzer 的一个用例。关于 model analyzer 参数和运行选项的更完整说明，请参见[这份](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html)指南。

*请注意，性能和模型分析实验均在配备 Intel i9 CPU 和 NVIDIA A6000 GPU 的系统上完成。实际结果可能因硬件不同而有所差异。*

## 自定义

`model.py` 文件特意保持精简，以最大化通用性。如果你想修改 transformer 模型的行为，例如增加返回的生成序列数量，请务必修改对应的 `config.pbtxt` 和 `model.py` 文件，再复制进 `model_repository`。

本教程使用的 transformers 模型都适合文本生成任务，但这并非限制。本教程的原理同样适用于服务任何其他 transformer 任务的模型。

Triton 还提供了本教程未提及的大量服务器配置选项。如需更定制化的部署，请参考我们的[模型配置指南](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)，了解如何在本教程的基础上扩展以满足你的需求。
