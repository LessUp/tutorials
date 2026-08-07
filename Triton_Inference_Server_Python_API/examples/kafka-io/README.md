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

# Triton Inference Server Kafka I/O 部署

借助 Triton Inference Server 进程内 Python API，你可以把基于 Triton 的模型集成到任意 Python 框架中，从 Kafka 主题（topic）消费消息，并把推理结果写回指定的 Kafka 主题。

本目录包含一个基于 Kafka I/O 的 Triton Inference Server 部署示例：服务器、消费者、生产者各自运行在一个独立的线程中。

> 💡 **AI Infra 视角**：在这个架构里，Triton 的角色是纯粹的推理引擎（inference engine），Kafka 则承担消息管道：生产者和消费者通过主题解耦，彼此无需知道对方的存在。这种「消息队列 + 推理引擎」的组合非常适合异步场景和削峰填谷——请求洪峰到来时 Kafka 先把流量兜住，Triton 按自己的节奏消费，不会被打爆。

| [安装](#安装) | [运行部署](#启动流水线) | [发送请求](#向部署发送请求) |


## 安装

在这条 Kafka I/O 流水线中，我们部署了一个基于 `transformers` tokenization 模块的预处理阶段（tokenization），该方案可按需扩展到任意类型的模型。

> 💡 **AI Infra 视角**：这里的 tokenizer 以自定义模型的形式跑在 Triton 的 Python backend 里。生产实践中，「前后处理（pre/post-processing）逻辑放在推理引擎内，还是放在编排层」是一道常见的架构选择题：放在引擎内能贴近计算、减少数据搬运，但会让模型仓库的职责变重。本例把分词当作「模型」来部署，演示的正是前一种思路。

### 前置条件
1. [Docker](https://docs.docker.com/engine/install/)

### 启动 Docker 容器
docker 服务启动后，执行以下命令启动一个容器：

```bash
docker run --rm -it --gpus all -v <path>/<to>/tutorials/Triton_Inference_Server_Python_API/examples/kafka-io/:/opt/tritonserver/kafka-io -w /opt/tritonserver/kafka-io  --entrypoint bash nvcr.io/nvidia/tritonserver:26.07-py3
```

### Clone Repository

```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Triton_Inference_Server_Python_API/examples/kafka-io
```

*注意：如果你已经将本地目录中的 git 仓库挂载进 Docker 容器，可以跳过这一步*


### 安装依赖

请注意，安装耗时取决于你的硬件配置和网络连接。


```bash
pip install -r requirements.txt
```

如果尚未安装 Triton server，请使用以下命令安装依赖。

```bash
pip install /opt/tritonserver/python/tritonserver-2.44.0-py3-none-any.whl
```

接下来运行提供的 `start-kafka.sh` 脚本，它会依次完成以下工作：
1. 下载 Kafka 及其依赖
2. 通过启动 Zookeeper 和 Kafka broker 来启动 Kafka 服务
3. 创建两个主题：`inference-input` 作为输入队列，`inference-output` 用于存放推理结果

```bash
chmod +x start-kafka.sh
./start-kafka.sh
```

## 启动流水线

### 启动推理流水线

运行提供的 `start-server.sh` 脚本，它会依次完成以下工作：
1. 导出 Kafka 生产者和消费者配置、输入/输出主题名称、模型名称与模型仓库路径。
2. 启动服务器。

```bash
chmod +x start-server.sh
./start-server.sh
```

当控制台输出类似下面这样的日志时：
```bash
2024-07-18 21:55:38,254 INFO api.py:609 -- Deployed app 'default' successfully.
```
说明服务器已成功启动。此时可以按 `Ctrl+C` 退出，进入下一步。

> 💡 **AI Infra 视角**：`KAFKA_CONSUMER_MAX_WORKER_THREADS` 控制消费者的工作线程数。Kafka 中同一消费组（consumer group）内一个分区（partition）同一时刻只能被一个消费者读取，所以真正提升吞吐的路径通常是增加分区数、再水平扩展消费者；单机线程池只是初步调优手段。这是「消息队列 + 推理服务」架构里最常见的性能瓶颈点，值得优先排查。

*注意：上面的调用默认使用 1 个线程跑 Kafka 消费者；如需提高并发度，请将环境变量 `KAFKA_CONSUMER_MAX_WORKER_THREADS` 设为期望值并重启服务器。重启后消费者将以新的并发度运行，从而提高整个部署的吞吐。*

## 向部署发送请求

要向已部署的推理流水线发送请求，请使用以下命令向输入 Kafka 主题写入消息。

```bash
cd kafka_2.13-3.7.0
bin/kafka-console-producer.sh --topic inference-input --bootstrap-server localhost:9092
```

命令执行后，你会看到 `>` 提示符，可以开始向输入主题写入消息。

```bash
> this is a sample message
>
```

写入足够多的消息后，按 `Ctrl+C` 即可退出提示符。

#### 示例输出
工作流从 Kafka 主题消费消息后，会调用 Triton server，并把推理输出以 `json` 字符串的形式写入输出 Kafka 主题。消息被消费后，我们可以启动消费者查看流水线写入输出主题的消息。

```bash
bin/kafka-console-consumer.sh --topic inference-output --from-beginning --bootstrap-server localhost:9092
```

由于本示例在 Triton 中部署了一个 tokenizer 自定义模型，我们应该能在 Kafka 主题中看到如下所示的输出。

```bash
{"model": {"name": "tokenizer", "version": 1, "state": null, "reason": null}, "request_id": "", "parameters": {}, "outputs": {"input_ids": [[101, 1142, 1110, 2774, 3802, 118, 1207, 130, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, "error": null, "classification_label": null, "final": true}
```
