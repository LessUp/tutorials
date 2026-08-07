<!---
# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
--->

# 使用 Triton Server 和 TensorRT-LLM 的多节点生成式 AI

大语言模型（LLM）很"大"，这几乎是不言自明的。
LLM 往往大到单块 GPU 的内存装不下。
因此，我们需要一种方案，让多块 GPU 协同工作，为这些超大型模型提供推理服务。

本指南旨在讲解如何使用 Triton Server 和 TRT-LLM 在 Kubernetes 集群中部署多 GPU、多节点的 LLM。
使用 Triton Inference Server、TensorRT-LLM 和 Kubernetes 搭建多节点 LLM 支持并不困难，但确实需要事先做好准备。

我们将涵盖以下主题：

* [集群搭建（Cluster Setup）](#cluster-setup)
  * [持久卷设置（Persistent Volume Setup）](#persistent-volume-setup)
  * [核心集群服务（Core Cluster Services）](#core-cluster-services)
    * [Kubernetes Node Feature Discovery 服务](#kubernetes-node-feature-discovery-service)
    * [Kubernetes 的 NVIDIA Device Plugin](#nvidia-device-plugin-for-kubernetes)
    * [NVIDIA GPU Feature Discovery 服务](#nvidia-gpu-feature-discovery-service)
  * [Hugging Face 授权](#hugging-face-authorization)
* [Triton 准备（Triton Preparation）](#triton-preparation)
  * [模型准备脚本](#model-preparation-script)
  * [自定义容器镜像](#custom-container-image)
  * [Kubernetes 拉取凭据（Pull Secrets）](#kubernetes-pull-secrets)
* [Triton 部署（Triton Deployment）](#triton-deployment)
  * [工作原理](#how-it-works)
  * [潜在的改进方向](#potential-improvements)
    * [自动扩缩容与组调度（Gang Scheduling）](#autoscaling-and-gang-scheduling)
    * [网络拓扑感知调度](#network-topology-aware-scheduling)
* [本指南的开发过程（Developing this Guide）](#developing-this-guide)

开始本指南/教程之前，你需要准备以下几样东西。

* Kubernetes 控制 CLI（`kubectl`）
  [ [文档](https://kubernetes.io/docs/reference/kubectl/introduction/)
  | [下载](https://kubernetes.io/releases/download/) ]
* Helm CLI（`helm`）
  [ [文档](https://helm.sh/)
  | [下载](https://helm.sh/docs/intro/install) ]
* Docker CLI（`docker`）
  [ [文档](https://docs.docker.com/)
  | [下载](https://docs.docker.com/get-docker/) ]
* 合适的文本编辑工具，用于编辑 YAML 文件。
* Kubernetes 集群。
* 配置完成、且对集群拥有管理员权限的 `kubectl`。

## 集群搭建（Cluster Setup）

以下说明用于配置一个 Kubernetes 集群，使用 Triton Server 和 TRT-LLM 部署 LLM。

### 前置条件

本指南假定所有带 NVIDIA GPU 的节点都已具备以下配置：
- 节点标签 `nvidia.com/gpu=present`，用于更方便地识别带有 NVIDIA GPU 的节点。
- 节点污点 `nvidia.com/gpu=present:NoSchedule`，用于阻止非 GPU 的 Pod 被调度到 GPU 节点上。

> [!Tip]
> 如果使用 AKS、EKS 或 GKE 这类 Kubernetes 托管服务，配置节点时通常最好使用它们各自的控制界面，而不是直接用 `kubectl` 操作。

### 持久卷设置（Persistent Volume Setup）

为了让部署在不同节点上的多个 Pod 能够加载同一个模型的不同分片（shard），从而协同服务那些单块 GPU 无法承载的超大推理请求，我们需要一个共享存储位置。
在 Kubernetes 中，这种共享存储位置称为持久卷（persistent volume）。
持久卷可以同时映射挂载到任意数量的 Pod 中，Pod 内的进程可以像访问自己文件系统的一部分那样访问它。

此外，我们还需要创建一个持久卷声明（persistent-volume claim，PVC），用它把持久卷分配给某个 Pod。

遗憾的是，持久卷的创建方式取决于集群的搭建方式，这超出了本教程的范围。
不过，我们会提供一个基本流程概述。

> 💡 **AI Infra 视角**：多节点推理的"共享存储"是刚需：每个 rank 要加载同一份模型权重分片和 tokenizer，如果各节点各自下载或生成，既慢又浪费带宽。把模型仓库放到共享存储上，一份文件多处挂载，模型更新只需替换一份文件，这也是 AI 推理集群的常见做法。注意存储性能（尤其 EFS 的吞吐上限）在模型并发加载时会成为瓶颈，生产环境常改用 FSx for Lustre 这类高性能并行文件系统。

#### 创建持久卷

如果你的集群由云服务商（CSP）托管，例如 Amazon（EKS）、Azure（AKS）或 gCloud（GKE），网上都有为集群设置持久卷的分步教程。
否则，你需要与集群管理员合作，或者在网上另找一份为集群设置持久卷的指南。

以下资源可以帮助你为集群设置持久卷。

* [Kubernetes 持久卷](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
* [AKS 持久卷](https://learn.microsoft.com/en-us/azure/aks/azure-csi-disk-storage-provision)
* [EKS 持久卷](https://aws.amazon.com/blogs/storage/persistent-storage-for-kubernetes/)
* [GKE 持久卷](https://cloud.google.com/kubernetes-engine/docs/concepts/persistent-volumes)
* [OKE 持久卷](https://docs.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcreatingpersistentvolumeclaim.htm)

> [!Important]
> 考虑集群将要承载的模型的存储需求非常重要，请务必让持久卷的容量足以容纳所有模型的总存储大小。

下面是一些本教程内部测试时得到的示例值。

| 模型           | 并行度 | 原始大小 | 转换后大小 | 总计大小 |
| :-------------- | ----------: | -------: | -------------: | ---------: |
| **Llama-3-8B**  | 2           | 15Gi     | 32Gi           | 47Gi       |
| **Llama-3-8B**  | 4           | 15Gi     | 36Gi           | 51Gi       |
| **Llama-3-70B** | 8           | 90Gi     | 300Gi          | 390Gi      |

> [!Important]
> 注意表中"转换后大小"远大于原始大小——TRT-LLM 引擎文件比原始权重更"重"，其中包含算子融合后的 kernel、内存布局规划等。规划存储时最容易低估的就是这一点：转换后的引擎文件通常接近甚至数倍于原始模型大小，而且每个"并行度配置 × GPU 型号"组合都需要单独一份引擎。这也是生产环境要做存储容量预算和引擎缓存清理策略的原因。

#### 创建持久卷声明

要把 Triton Server Pod 连接到上面创建的持久卷，我们需要创建一个持久卷声明（PVC）。你可以使用本教程提供的 [pvc.yaml](./pvc.yaml) 文件来创建。

> [!Important]
> `volumeName` 属性必须与上面创建的持久卷的 `metadata.name` 属性一致。

### 核心集群服务（Core Cluster Services）

所有节点正确打标签和设置污点后，按照以下步骤准备集群，使用 Triton Server 跨多个节点部署大语言模型。

下面一系列步骤是为全新集群准备的。
对于处于各种中间状态的集群，最好先与集群管理员协调，再安装新的服务和能力。

#### Kubernetes Node Feature Discovery 服务

1.  将 Kubernetes Node Feature Discovery chart 仓库添加到本地缓存。

    ```bash
    helm repo add kube-nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts \
      && helm repo update
    ```

2.  运行以下命令安装该服务。

    ```bash
    helm install -n kube-system node-feature-discovery kube-nfd/node-feature-discovery \
      --set nameOverride=node-feature-discovery \
      --set worker.tolerations[0].key=nvidia.com/gpu \
      --set worker.tolerations[0].operator=Exists \
      --set worker.tolerations[0].effect=NoSchedule
    ```

    > [!Note]
    > 上面的命令设置了容忍（toleration）值，允许 Pod 被调度到带有匹配污点的节点上。
    > 参见本文档的[前置条件](#prerequisites)，了解本文档预期已应用到集群 GPU 节点上的污点。

#### Kubernetes 的 NVIDIA Device Plugin

1.  如果你的集群已经安装了 Device Plugin，可以跳过这一步。
    AKS、EKS 和 GKE 这类云厂商的托管 Kubernetes 集群，通常在向集群添加 GPU 节点时就会自动安装 Device Plugin。

    要检查你的集群是否需要 Kubernetes 的 NVIDIA Device Plugin，请运行以下命令，并在输出中查找 `nvidia-device-plugin-daemonset`。

    ```bash
    kubectl get daemonsets --all-namespaces
    ```

    示例输出：
    ```text
    NAMESPACE     NAME         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
    kube-system   kube-proxy   6         6         6       6            6
    ```

2.  如果列表中没有 `nvidia-device-plugin-daemonset`，请运行下面的命令安装该插件。
    安装后，它会让容器能够访问集群中的 GPU。

    更多信息参见
    [Github/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin/blob/main/README.md)。

    ```bash
    kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
    ```

#### NVIDIA GPU Feature Discovery 服务

1.  如果你的集群已经安装了该服务，可以跳过这一步。

    要检查你的集群是否需要 Kubernetes 的 NVIDIA Device Plugin，请运行以下命令，并在输出中查找 `gpu-feature-discovery`。

    ```bash
    kubectl get daemonsets --all-namespaces
    ```

    示例输出：
    ```text
    NAMESPACE     NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
    kube-system   kube-proxy                       6         6         6       6            6
    kube-system   nvidia-device-plugin-daemonset   6         6         6       6            6
    ```

2.  如果列表中已有 `gpu-feature-discovery`，请跳过这一步和下一步。

    否则，请使用下面的 YAML 文件安装 GPU Feature Discovery 服务。

    > [nvidia_gpu-feature-discovery_daemonset.yaml](nvidia_gpu-feature-discovery_daemonset.yaml)

    上面的文件是从
    [GitHub/NVIDIA](https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.8.2/deployments/static/gpu-feature-discovery-daemonset.yaml)
    下载内容，并针对本教程做了修改。

    ```bash
    curl https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.8.2/deployments/static/gpu-feature-discovery-daemonset.yaml \
      >  nvidia_gpu-feature-discovery_daemonset.yaml
    ```

3.  然后运行以下命令进行安装。

    ```bash
    kubectl apply -f ./nvidia_gpu-feature-discovery_daemonset.yaml
    ```

> 💡 **AI Infra 视角**：这一组"核心集群服务"（NFD → Device Plugin → GFD）层层递进：NFD 探测节点硬件特性，Device Plugin 把 GPU 变成可调度资源，GFD 再把 GPU 型号等特性打成节点标签。三者配合，K8s 调度器才能做到"按 GPU 数量调度 + 按 GPU 型号选节点"。多节点模型部署时，这些组件更是前提——调度器要精确地把整组 Pod 放到互连最优的节点上。

### Hugging Face 授权

要从 Hugging Face 下载模型，你的 Pod 需要一个具有相应权限的访问令牌（access token）才能从他们的服务器下载模型。

1.  如果你还没有 Hugging Face 访问令牌，需要创建一个。
    创建方法参见
    [他们的指南](https://huggingface.co/docs/hub/en/security-tokens)。

2.  拿到令牌后，用下面的命令把它保存为集群中名为 `hf-model-pull` 的 secret。

    ```bash
    kubectl create secret generic hf-model-pull '--from-literal=password=<access_token>'
    ```

3.  要验证 secret 是否创建成功，使用下面的命令并在输出中查找你的 secret。

    ```bash
    kubectl get secrets
    ```

## Triton 准备（Triton Preparation）

### 模型准备脚本

这个脚本的职责是从 Hugging Face 获取模型文件、生成 TensorRT 引擎和 plan 文件，并缓存这些生成的文件。
脚本依赖这样一个事实：我们要用的 Kubernetes 部署脚本依赖持久卷，而持久卷由 Helm chart 中提供的持久卷声明（PVC）支撑。

具体来说，模型目录和引擎目录会映射到持久卷上的文件夹，并重新映射到 Helm chart 部署的所有后续 Pod 中。
这样生成脚本就能检测到 plan 和引擎的生成步骤已完成，从而避免重复劳动。

> [!Tip]
> 每次安装 Helm chart 时，这个脚本都会作为一个 Job 执行，除非把 `.model.skipConversion` 属性设为 `true`。

Triton Server 启动时，相同的持久卷文件夹会被挂载到它的容器中，Triton 会直接使用预生成的模型 plan 和引擎文件。
这不仅让不同节点上的 Pod 可以共享同一份模型引擎和 plan 文件，还大幅缩短了同一节点上后续 Pod 的启动时间。

> [!Note]
> 你可以在 [containers/server.py](containers/server.py) 中查看获取和转换模型的代码。
> 这个文件在构建镜像时被复制进服务器容器镜像（见下文），然后在转换 Job 的 Pod 部署时执行。

#### 自定义容器镜像

1.  使用下面的文件，我们在下一步创建一个自定义容器镜像。

    > [triton_trt-llm.containerfile](containers/triton_trt-llm.containerfile)

2.  运行下面的命令创建一个自定义的 Triton Inference Server 镜像，包含生成 TensorRT-LLM plan 和 engine 文件所需的全部工具。本例使用标签 `24.04`，与基础镜像的 `24.04-trtllm-python-py3` 日期部分保持一致。

    ```bash
    docker build \
      --file ./triton_trt-llm.containerfile \
      --rm \
      --tag triton_trt-llm:24.04 \
      .
    ```

    ##### Triton CLI 的自定义版本

    这个自定义的 Triton Server 容器镜像使用了一个自定义版本的 Triton CLI。
    相关改动已作为
    [topic 分支](https://github.com/triton-inference-server/triton_cli/tree/jwyman/aslb-mn) 发布在 GitHub 上的 Triton CLI 仓库中。
    该分支的改动可以通过 GitHub 界面[查看](https://github.com/triton-inference-server/triton_cli/compare/main...jwyman/aslb-mn)，主要包含：为 TensorRT-LLM 优化模型时支持指定张量并行度，以及支持更多模型。

3.  将容器镜像上传到集群可访问的仓库。

    要让 Kubernetes 集群能够下载我们的新容器镜像，必须把它推送到集群节点可以访问的容器镜像仓库。
    本例使用虚构的 `nvcr.io/example` 仓库做演示。
    你需要确定自己有哪些可写权限的仓库，并且你的集群也能访问这些仓库。

    1. 首先，像下面这样用仓库名重新标记容器镜像。

        ```bash
        docker tag \
          triton_trt-llm:24.04 \
          nvcr.io/example/triton_trt-llm:24.04
        ```

    2. 接下来，把容器镜像上传到你的仓库。

        ```bash
        docker push nvcr.io/example/triton_trt-llm:24.04
        ```

#### Kubernetes 拉取凭据（Pull Secrets）

如果你的容器镜像仓库要求凭据才能拉取镜像，那么你需要创建一个 Kubernetes docker-registry 类型的 secret。
下面用上面的 `nvcr.io` 容器镜像仓库示例做演示。
请务必正确转义密码或用户名中的特殊字符，例如 `$`。

1.  使用下面的命令创建所需的 secret。你的仓库对应的 secret 应该与下面的示例类似，但不会完全相同。

    ```bash
    kubectl create secret docker-registry ngc-container-pull \
      --docker-password='dGhpcyBpcyBub3QgYSByZWFsIHNlY3JldC4gaXQgaXMgb25seSBmb3IgZGVtb25zdHJhdGlvbiBwdXJwb3Nlcy4=' \
      --docker-server='nvcr.io' \
      --docker-username='\$oauthtoken'
    ```

2.  上面的命令会在你的集群中创建一个名为 `ngc-container-pull` 的 secret。
    你可以用下面的命令验证 secret 是否创建正确，并在输出中查找对应的 secret。

    ```bash
    kubectl get secrets
    ```

3.  要确认 secret 的内容正确，可以运行下面的命令。

    ```bash
    kubectl get secret/ngc-container-pull -o yaml
    ```

    你应该会看到类似下面的输出。

    ```yaml
    apiVersion: v1
    data:
      .dockerconfigjson: eyJhdXRocyI6eyJudmNyLmlvIjp7InVzZXJuYW1lIjoiJG9hdXRodG9rZW4iLCJwYXNzd29yZCI6IlZHaHBjeUJwY3lCdWIzUWdZU0J5WldGc0lITmxZM0psZEN3Z2FYUWdhWE1nYjI1c2VTQm1iM0lnWkdWdGIyNXpkSEpoZEdsdmJpQndkWEp3YjNObGN5ND0iLCJhdXRoIjoiSkc5aGRYUm9kRzlyWlc0NlZrZG9jR041UW5CamVVSjFZak5SWjFsVFFubGFWMFp6U1VoT2JGa3pTbXhrUTNkbllWaFJaMkZZVFdkaU1qVnpaVk5DYldJelNXZGFSMVowWWpJMWVtUklTbWhrUjJ4MlltbENkMlJZU25kaU0wNXNZM2swWjFWSGVHeFpXRTVzU1VjMWJHUnRWbmxKU0ZaNldsTkNRMWxZVG14T2FsRm5aRWM0WjJGSGJHdGFVMEo1V2xkR2MwbElUbXhaTTBwc1pFaE5hQT09In19fQ==
    kind: Secret
    metadata:
      name: ngc-container-pull
      namespace: default
    type: kubernetes.io/dockerconfigjson
    ```

    `.dockerconfigjson` 的值是一个 base-64 编码的字符串，可以解码成下面的内容。

    ```json
    {
      "auths": {
        "nvcr.io": {
          "username":"$oauthtoken",
          "password":"VGhpcyBpcyBub3QgYSByZWFsIHNlY3JldCwgaXQgaXMgb25seSBmb3IgZGVtb25zdHJhdGlvbiBwdXJwb3Nlcy4gUGxlYXNlIG5ldmVyIHVzZSBCYXNlNjQgdG8gaGlkZSByZWFsIHNlY3JldHMh",
          "auth":"JG9hdXRodG9rZW46VkdocGN5QnBjeUJ1YjNRZ1lTQnlaV0ZzSUhObFkzSmxkQ3dnYVhRZ2FYTWdiMjVzZVNCbWIzSWdaR1Z0YjI1emRISmhkR2x2YmlCd2RYSndiM05sY3k0Z1VHeGxZWE5sSUc1bGRtVnlJSFZ6WlNCQ1lYTmxOalFnZEc4Z2FHbGtaU0J5WldGc0lITmxZM0psZEhNaA=="
        }
      }
    }
    ```

    你可以用下面这一行紧凑命令直接得到上面的输出。

    ```bash
    kubectl get secret/ngc-container-pull -o json | jq -r '.data[".dockerconfigjson"]' | base64 -d | jq
    ```

    > [!Note]
    > `password` 和 `auth` 的值也是 base-64 编码的字符串。
    > 我们建议检查以下值：
    >
    > * `.auths['nvcr.io'].username` 的值。
    > * `.auths['nvcr.io'].password` 的 base64 解码值。
    > * `.auths['nvcr.io'].auth` 的 base64 解码值。

## Triton 部署（Triton Deployment）

> [!Note]
> 部署一个能装进单块 GPU 的模型到 Triton Server 上很简单，但本指南不讲解这部分。
> 单 GPU 或单节点多 GPU 的模型部署说明和示例，请改用
> [使用 Triton Server 和 TensorRT-LLM 对生成式 AI 进行自动扩缩容与负载均衡指南](../TensorRT-LLM_Autoscaling_and_Load_Balancing/README.md)。

考虑到某些 AI 模型的内存需求，单块设备无法承载它们。
Triton 和 TensorRT-LLM 提供了一种机制，让多个 GPU 设备协同工作来承载大模型。
提供的示例 Helm [chart](./chart/) 就提供了利用这一能力的机制。

要启用该功能，请把 `model.tensorrtLlm.parallelism.tensor` 的值改为大于 1 的整数。
为模型配置张量并行（tensor parallelism）后，TensorRT-LLM 运行时会高效地合并多块 GPU 的内存，从而承载单块 GPU 装不下的模型。

同样，修改 `model.tensorrtLlm.parallelism.pipeline` 的值可以启用流水线并行（pipeline parallelism）。
流水线并行用于合并多块 GPU 的计算能力，并行处理推理请求。

> [!Important]
> `.tensor` 和 `.pipeline` 值的乘积应为大于 `0` 且小于或等于 `32` 的 2 的幂。

承载模型所需的 GPU 数量等于 `.tensor` 和 `.pipeline` 值的乘积。
部署模型时，每块所需 GPU 会创建一个 Pod。
Helm chart 会创建一个 leader Pod，并根据承载模型还需要多少个 Pod，创建一个或多个 worker Pod。
此外，还会创建一个模型转换 Job，用于从 Hugging Face 下载模型，并把下载的模型转换成 TRT-LLM 引擎和 plan 文件。
要禁用 Helm chart 创建转换 Job，请把 values 文件中的 `model.skipConversion` 属性设为 `true`。

> [!Warning]
> 如果你的集群资源不足，无法同时创建转换 Job、leader Pod 和所需的 worker Pod，而 Job Pod 又没有被优先调度执行，示例 Helm chart 可能会"卡住"——因为 leader Pod 在等待 Job Pod 完成，而资源不足导致 Job Pod 无法被调度。
>
> 如果发生这种情况，最好删除 Helm 安装并重试，直到 Job Pod 被成功调度。
> Job Pod 完成后会释放资源，让其他 Pod 得以启动。

> 💡 **AI Infra 视角**：这里埋着一个多节点推理的经典资源调度坑：leader 等待转换 Job 完成，而 Job 又因资源不足排不上队，形成互相等待的死锁。生产上的对策是资源配额预留（给 Job 单独划资源池）、优先级（PriorityClass）让 Job 先跑，或干脆把模型转换放到集群外的 CI/CD 流水线里完成，集群里只做推理。

### 部署单 GPU 模型

按照下面的步骤，部署一个能装进单块 GPU 的模型到 Triton Server 上，非常简单直接。

1.  创建一个包含必需值的自定义 values 文件：

    * 容器镜像名称。
    * 模型名称。
    * 支持/可用的 GPU。
    * 镜像拉取凭据（如有必要）。
    * Hugging Face secret 名称。

    提供的示例 Helm [chart](./chart/) 中包含几个示例 values 文件，例如
    [llama-3-8b_values.yaml](chart/llama-3-8b-instruct_values.yaml)。

2.  在 Triton + TRT-LLM 上部署 LLM。

    用下面的命令应用自定义 values 文件来覆盖导出的基础 values 文件，并创建 Triton Server Kubernetes deployment。

    > [!Tip]
    > 命令行上指定 values 文件的顺序很重要，values 会按指定的顺序依次应用并覆盖已有值。

    ```bash
    helm install <installation_name> \
      --values ./chart/values.yaml \
      --values ./chart/<custom_values>.yaml \
      --set 'triton.image.name=<custom_image_name>' \
      ./chart/.
    ```

    > [!Important]
    > 请务必把上面示例中的 `<installation_name>` 和 `<custom_values>` 替换成正确的值。

3.  验证 Chart 安装。

    使用下面的命令检查已安装的 chart，确认一切是否按预期工作。

    ```bash
    kubectl get deployments,pods,services,jobs --selector='app=<installation_name>'
    ```

    > [!Important]
    > 请务必把上面示例中的 `<installation_name>` 替换成正确的值。

    输出应该类似下面这样（假设安装名称为 "llama-3"）：

    ```text
    NAME                      READY   UP-TO-DATE   AVAILABLE
    deployment.apps/llama-3   0/1     1            0

    NAME                          READY   STATUS    RESTARTS
    pod/llama-3-7989ffd8d-ck62t   0/1     Pending   0

    NAME              TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)
    service/llama-3   ClusterIP   10.100.23.237   <none>        8000/TCP,8001/TCP,8002/TCP
    ```

4.  卸载 Chart

    卸载 Helm chart 很简单，运行下面的命令即可。
    这在尝试各种选项和配置时非常有用。

    ```bash
    helm uninstall <installation_name>
    ```

### 工作原理

Helm chart 会创建一个模型转换 Job 和多个 Kubernetes deployment，以支撑分布式模型的张量并行需求。
部署分布式模型时，会创建一个 "leader" Pod，以及满足模型张量并行需求所需的一定数量 "worker" Pod。
leader Pod 随后等待转换 Job 完成，并等待所有 worker Pod 成功部署。

模型转换 Job 负责从 Hugging Face 下载配置好的模型，并把模型转换成一套 TensorRT-LLM 可用的引擎和 plan 文件。
模型转换 Job 会把所有下载和转换的文件放到提供的持久卷上。

> [!Note]
> 从 Hugging Face 下载的模型在可能的情况下会被复用。
> 转换后的 TRT-LLM 模型与 GPU 型号和张量并行配置强相关。
> 因此，模型部署到每一种 GPU 以及每一种张量并行配置，都会各存在一份转换后的模型。

上述条件满足后，leader Pod 会启动一个 [`mpirun`](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html) 进程，为分布式模型的每个 Pod 各创建一个 Triton Server 进程。

leader Pod 的进程负责推理请求与响应的收发功能，以及推理请求的 token 化和结果的去 token 化。
worker Pod 的进程提供扩展的 GPU 计算和内存能力。
所有进程由最初的 `mpirun` 进程统一协调。
进程之间的通信由 [NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl)（NCCL）加速。
NCCL 支持 GPU 到 GPU 的直接通信，避免了原本 GPU→CPU→GPU 的数据拷贝浪费。

> 💡 **AI Infra 视角**：这是多节点推理的标准架构：一个 leader 对外提供 HTTP/gRPC 入口（含 tokenizer 前后处理），多个 rank 通过 MPI 拉起、通过 NCCL 做集合通信，对外表现为"一个逻辑模型"。NCCL 的 AllReduce 是张量并行每层都要做的通信，跨节点时走 RDMA（EFA/IB）才能把延迟压住；如果误走了 TCP，吞吐会断崖式下跌。上线前用 NCCL Test 验证跨节点带宽，是这套架构的必修课。

### 潜在的改进方向

#### 自动扩缩容与组调度（Gang Scheduling）

本指南没有提供 Triton 部署的自动扩缩容或负载均衡方案，因为 Kubernetes 的水平 Pod 自动扩缩（HPA）无法管理由多个 Pod 组成的部署。
此外，由于本教程的方案使用了多个 deployment，任何自动化都有很高的风险出现并发、部分部署的情况，耗尽可用资源，导致所有 deployment 都无法成功。

举一个并发部分部署互相阻塞的例子：假设一个集群有 4 个节点，每个节点 8 块 GPU，总共 32 块可用 GPU。
现在考虑一个需要 8 块 GPU 的模型，我们尝试部署它的 5 个副本。
逐个部署时，每个 deployment 都会分到 8 块 GPU，直到可用 GPU 变为 0，最终模型成功部署了 4 次。
此时系统知道没有更多可用资源，第 5 个副本部署失败。

然而，如果同时尝试部署全部 5 个副本，很可能每个副本都会至少分到 1 块 GPU。
结果至少有 2 个副本资源不足，导致这两个 deployment 都停留在不可用的部分部署状态。

解决这个问题的一个方案是利用 Kubernetes 的组调度器（gang scheduler）。
组调度让 Kubernetes 调度器只有在某个 Pod 的整个"同组"（cohort）Pod 都能被创建时，才创建这个 Pod。
这样就解决了模型 Pod 部分部署互相阻塞、无法完整部署的问题。

> [!Note]
> 更多组调度的信息，参见 [维基百科上的 gang scheduling](https://en.wikipedia.org/wiki/Gang_scheduling)。

不过，上述方案并没有提供任何自动扩缩容方案。
要实现自动扩缩容，需要一个支持组调度的自定义自动扩缩器。

> 💡 **AI Infra 视角**：组调度的本质是"全有或全无"——要么一整组 Pod 全部获得资源，要么一个都别调度。对多节点推理来说，部分调度的后果是 NCCL 初始化卡死、资源白白占用。文中 5 副本的例子正是经典的资源碎片死锁：每个副本分到一点 GPU，谁都不够跑，互相锁死。生产上要么用 gang scheduler（如 Volcano、Kueue 的 cohort 语义），要么用配额准入控制。

#### 网络拓扑感知调度

Triton Server + TensorRT-LLM 利用高度优化的网络栈
[NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl)（NCCL）来实现张量并行。
NCCL 利用现代 GPU 支持
[远程直接内存访问](https://en.wikipedia.org/wiki/Remote_direct_memory_access)（RDMA）网络加速的能力，优化 GPU 之间的操作，无论它们在同一台机器上还是在相邻的机器上。
这意味着，不同机器上的 GPU 之间的网络质量直接决定分布式模型的性能。

为 Kubernetes 提供网络拓扑感知的调度器，有助于确保分配给某个模型部署各 Pod 的 GPU 相对彼此在拓扑上比较"近"。
理想情况下，位于同一台机器上，或者至少位于同一个网络交换机下，以尽量降低网络延迟和带宽限制的影响。

> 💡 **AI Infra 视角**：张量并行的通信量随 GPU 数量平方级增长，节点间网络质量直接决定模型能跑多快。拓扑感知调度（topology-aware scheduling）把"GPU 在拓扑上够不够近"（同一节点 > 同一机架 > 同一交换机）纳入调度决策，避免把跨节点 TP 的 Pod 拆到网络遥远的位置。云上对应物是 EFA 的 placement group，物理上把实例放到同一骨架交换机下，这是多节点推理性能优化的关键一环。

## 本指南的开发过程（Developing this Guide）

在编写本指南的过程中，我遇到了几个必须先解决的问题，然后才能写出一份有用的指南。
本节将概述我在开发过程中遇到的问题以及解决方法。

> _本文档是使用 Amazon EKS 提供的 Kubernetes 集群开发的。_
> _本地机房或 Azure AKS、GCloud GKE 等其他云厂商提供的集群可能需要对本文档做相应修改。_

### 为什么是这套软件组件？

本文档描述的一组软件包，几乎是"最小可用"的组合，不需要为每个包和依赖手写自定义 Helm chart 和 YAML 文件。
这是唯一能让这套方案工作的组件组合吗？
绝对不是，有很多替代方案也能满足需求。
这套组合只是我在本指南中碰巧选择的方案。

下面是本指南中每个软件包的用途概述。

#### Kubernetes 的 NVIDIA Device Plugin

让 GPU 能够被 Kubernetes 调度器当作资源处理是必需的。
没有这个组件，GPU 就无法被正确分配给容器。

#### Kubernetes 的 NVIDIA GPU Discovery 服务

根据节点上可用的 NVIDIA 设备和软件，自动为 Kubernetes 节点打标签。
没有这些标签，部署模型时就不可能指定具体的 GPU SKU，因为 Kubernetes 调度器把所有 GPU 都当作相同的（都用通用资源名 `nvidia.com/gpu` 引用）。

#### Kubernetes Node Discovery 服务

这是 [Kubernetes 的 NVIDIA GPU Discovery 服务](#nvidia-gpu-discovery-service-for-kubernetes) 的前提条件。

#### NVIDIA DCGM Exporter

为集群中 NVIDIA GPU 和其他设备提供硬件监控和指标。
没有它提供的指标，就无法监控 GPU 利用率、温度和其他指标。

虽然 Triton Server 本身也能采集和提供 NVIDIA 硬件指标，但依赖 Triton Server 提供这项服务有几个原因说明它并不理想。

首先，同一台机器上多个进程各自查询 NVIDIA 设备驱动获取当前状态、过滤出只与自己进程相关的值、再通过 Triton 的 open-metrics 服务器对外提供，其浪费程度与节点上超过第一个的 Triton Server 进程数量成正比。

其次，由于获取硬件指标需要与内核态驱动交互，查询会被串行化，给系统带来额外开销和延迟。

最后，从 Triton Server 采集指标的频率与从 DCGM Exporter 采集指标的频率是不同的。
把指标采集从 Triton Server 中分离出来，可以自定义采集频率，从而进一步降低节点上的进程开销。

##### 为什么 DCGM Exporter 的 values 文件是自定义的？

安装 DCGM Exporter Helm chart 时，我决定使用自定义 values 文件，有几个原因。

首先，我的专业观点是，集群中的每个容器都应该指定资源 limits 和 requests。
不这样做，会让节点暴露在多种与资源耗尽相关、且难以诊断的故障条件下。
内存不足错误是最明显也最容易定位的。
此外，当某个容器不受资源约束时，很容易发生难以复现的瞬时超时和时序错误（由 CPU 超卖引起），这会快速浪费整个工程团队的时间去分类、调试和解决。

其次，DCGM Exporter 进程在系统中找不到 NVIDIA 设备时会刷屏式地打印错误日志。
这主要是因为该服务最初是为非 Kubernetes 环境创建的。
因此我想限制 exporter 部署到哪些节点。
幸运的是，DCGM Helm chart 通过支持 node selector 选项让这件事变得很容易。

第三，由于带 NVIDIA GPU 的节点被打上了 `nvidia.com/gpu=present:NoSchedule` 污点，任何没有显式容忍该污点的 Pod 都无法被分配到这些节点，所以我需要给 DCGM Exporter Pod 添加对应的容忍（tolerations）。

最后，DCGM Exporter 的默认 Helm chart 缺少启动进程时通过命令行传入的 `--kubernetes=true` 选项。
没有这个选项，DCGM Exporter 就无法正确地把硬件指标与真正使用它的 Pod 关联起来，也就无法了解每个 Pod 如何使用分配给它的 GPU 资源。

### 为什么使用 Triton CLI 而不是 NVIDIA 提供的其他工具？

我选择使用新的 [Triton CLI](https://github.com/triton-inference-server/triton_cli) 工具来为 TensorRT-LLM 优化模型，而不是其他可用工具，原因有几个。

首先，使用 Triton CLI 把模型的转换和优化简化成了单条命令。

其次，依赖 Triton CLI 简化了容器的构建，因为所有依赖通过一条 `pip install` 命令就都满足了。

#### 为什么使用 Triton CLI 的自定义分支而不是官方发布版？

我决定使用 [Triton CLI 的自定义分支](https://github.com/triton-inference-server/triton_cli/tree/jwyman/aslb-mn)，因为本指南需要的一些功能在当时的任何官方发布版中都不存在。
该分支没有提 Merge Request，是因为添加所需功能的方式与维护者计划的改动方向不一致。
一旦我们达成一致，本指南将更新为使用官方发布版。

### 为什么 Chart 运行 Python 脚本而不是直接运行 Triton Server？

有两个原因：

1.  为了从 Hugging Face 获取模型、转换并优化成 TensorRT-LLM 格式、再在宿主机上缓存，我认为使用 [Pod 初始化容器](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) 是最直接了当的方案。

    为了充分利用初始化容器，我选择了使用新的 [Triton CLI](https://github.com/triton-inference-server/triton_cli) 工具的自定义 [server.py](./containers/server.py) 脚本。

2.  多 GPU 部署需要相当专用的命令行才能运行，而我不想用 Helm chart 脚本去生成它。
    利用自定义 Python 脚本是合理且最简单的方案。

#### 为什么 Python 代码写成那样？

因为我不是 Python 开发者，但我正在学！
我的背景是 C/C++，有大量 shell 脚本经验。

### 为什么使用自定义 Triton 镜像？

我决定使用自定义镜像，有几个原因。

1.  考虑到上面的答案以及 Triton CLI 和自定义 Python 脚本的使用，初始化容器需要预先装好这两个组件，以避免不必要的临时存储（ephemeral storage）使用。

    > [!Warning]
    > 使用临时存储可能导致 Pod 被驱逐（eviction），因此应尽可能避免。

2.  由于 Triton + TRT-LLM 镜像本身已经非常大，我不想再用另一个容器镜像占用额外的宿主机存储空间。

    此外，一个 Pod 在下载容器时看起来"卡在" `Pending` 状态，比"初始化容器前短暂 Pending、随后在 Triton Server 启动前长时间 Pending"更容易理解。

3.  我想要一个自定义的、固定的 `ENGINE_DEST_PATH` 环境变量，让初始化容器和 Triton Server 容器都能使用。

---

本文档涉及的软件版本：

* Triton Inference Server v2.45.0 (24.04-trtllm-python-py3)
* TensorRT-LLM v0.9.0
* Triton CLI v0.0.7
* Kubernetes 的 NVIDIA Device Plugin v0.15.0
* Kubernetes 的 NVIDIA GPU Discovery 服务 v0.8.2
* NVIDIA DCGM Exporter v3.3.5
* Kubernetes Node Discovery 服务 v0.15.4
* Kubernetes 的 Prometheus Stack v58.7.2
* Kubernetes 的 Prometheus Adapter v4.10.0

---

作者：J Wyman，系统软件架构师，AI 与分布式系统

Copyright &copy; 2024, NVIDIA CORPORATION. All rights reserved.
