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

# 在 EKS 上部署多节点 Triton + TRT-LLM

本仓库提供在 EKS（Amazon Elastic Kubernetes Service）上进行 LLM 多节点部署的完整指南，包括构建支持 EFA 等特性的自定义镜像、Helm chart 以及配套的 Python 脚本。该部署流程以 NVIDIA TensorRT-LLM 作为推理引擎，NVIDIA Triton Inference Server 作为模型服务器。

我们的部署方式是一个节点跑一个 Pod，因此多节点模型部署的核心难点在于：一个模型实例横跨多个节点，也就横跨了多个 Pod。这意味着"就绪后才可以接收请求"的原子单元以及扩缩容的基本单元，都变成了**一组 Pod**。本示例展示了如何解决这些问题，并提供了搭建以下功能的代码：

1. **用 LeaderWorkerSet 在 Pod 组上启动 Triton + TRT-LLM**：要在多个节点上启动 Triton 和 TRT-LLM，需要使用 MPI，让其中一个节点负责在所有节点（包括它自己）上拉起构成一个模型实例的 TRT-LLM 进程。这就要求我们提前知道所有相关节点的 hostname。因此，我们需要成组地创建 Pod，并且知道它们属于哪个模型实例组。为此我们使用 [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws/tree/main)，它可以创建由一组 Pod（1 个 leader Pod 加指定数量的 worker Pod）组成的"megapod"，并通过 Pod 标签标识组归属。LeaderWorkerSet 的配置以及通过 MPI 启动 Triton + TRT-LLM 的逻辑分别在 [`deployment.yaml`](multinode_helm_chart/chart/templates/deployment.yaml) 和 [server.py](multinode_helm_chart/containers/server.py) 中。

> 💡 **AI Infra 视角**：Kubernetes 的原生工作负载（Deployment、StatefulSet）都以单个 Pod 为调度和扩缩容单位，而多节点推理模型的"最小可用单元"是一整组 Pod——这正是需要 LeaderWorkerSet 这类新 API 的原因。它实际上是在 K8s 之上补了一层"组"的抽象：组内 Pod 一起创建、一起就绪、一起销毁，相当于给调度器传递了"这些 Pod 是一个整体"的语义。`1 个模型实例 = 1 个 megapod`，HPA 扩缩的也是这个 megapod。

2. **Gang Scheduling（组调度）**：组调度简单来说，就是保证构成一个模型实例的所有 Pod 全部就绪之后，才启动 Triton + TRT-LLM。我们在 [server.py](multinode_helm_chart/containers/server.py) 的 `wait_for_workers` 函数中演示了如何借助 `kubessh` 实现这一点。

> 💡 **AI Infra 视角**：组调度的本质是"全有或全无"——要么一整组 Pod 全部获得资源，要么一个都别调度。对多节点推理来说，如果只有部分节点被调度、其余的还在排队，先启动的进程会一直空等（比如 NCCL 初始化卡死），造成资源白白占用。这也是 LLM 训练/推理集群里常见的死锁场景：资源碎片化导致互相等待，最终谁也起不来。生产上要么用组调度器（gang scheduler），要么用准入控制（admission control）保证资源配额。

3. **自动扩缩容（Autoscaling）**：默认情况下，Horizontal Pod Autoscaler（HPA）只扩缩单个 Pod，而 LeaderWorkerSet 让"megapod"成为可扩缩对象。不过，由于这是 GPU 工作负载，我们不想用 CPU 和主机内存用量来做扩缩容判断。我们演示了如何利用 Triton Server 通过 Prometheus 暴露的指标，在 [`triton-metrics_prometheus-rule.yaml`](multinode_helm_chart/triton-metrics_prometheus-rule.yaml) 中配置 GPU 利用率相关的记录规则（recording rule）。我们还演示了如何在 [`pod-monitor.yaml`](multinode_helm_chart/chart/templates/pod-monitor.yaml) 和 [`hpa.yaml`](multinode_helm_chart/chart/templates/hpa.yaml) 中正确配置 PodMonitor 和 HPA（关键在于只从 leader Pod 采集指标）。Prometheus 的正确安装以及 GPU 指标的暴露方法见 [Configure EKS Cluster and Install Dependencies](./2.%20Configure_EKS_Cluster.md)。为了让部署能够在 HPA 触发扩容时动态添加节点，我们还配置了 [Cluster Autoscaler](./2.%20Configure_EKS_Cluster.md#10-install-cluster-autoscaler)。

> 💡 **AI Infra 视角**：推理扩缩容有两个层次：HPA 负责在**已有节点**内增减 Pod 实例，Cluster Autoscaler 负责在**节点层面**增删机器。两者配合才能形成完整的弹性闭环——HPA 发现 Pod 调度不下（资源不足）时，Cluster Autoscaler 检测到 `unschedulable` 的 Pod 就会加节点。对 GPU 集群而言，只给 HPA 配 CPU 指标是常见误区：GPU 服务可能 CPU 占用很低但队列已经积压，所以这里选用 Triton 的队列/计算时间比这类业务指标。

4. **LoadBalancer 设置**：虽然模型实例内存在多个 Pod，但每组中只有一个 Pod 接收请求。我们在 [`service.yaml`](multinode_helm_chart/chart/templates/service.yaml) 中展示了如何正确配置 LoadBalancer Service，让外部客户端能够提交请求。

> 💡 **AI Infra 视角**：多节点分布式模型对外只暴露一个入口（leader 的 HTTP/gRPC 端口），其余 rank 只做计算。这与"多个独立副本 + Service 做负载均衡"的架构不同——后者每个副本都能独立服务请求，前者只有一个逻辑服务点。理解这一点对排查"为什么 Service 后面那么多 Pod 只有 1 个在响应"很有帮助：它们是同一个模型的不同 rank，不是可独立服务的副本。

## 安装与配置

1. [创建 EKS 集群](1.%20Create_EKS_Cluster.md)
2. [配置 EKS 集群](2.%20Configure_EKS_Cluster.md)
3. [部署 Triton](3.%20Deploy_Triton.md)
