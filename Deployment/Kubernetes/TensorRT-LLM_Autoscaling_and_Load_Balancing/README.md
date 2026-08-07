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

# 使用 Triton Server 和 TensorRT-LLM 对生成式 AI 进行自动扩缩容与负载均衡

为 Triton Inference Server 服务的大型语言模型配置自动扩缩容（autoscaling）和负载均衡（load balancing）并不困难，但确实需要事先做好准备。

本指南概述了从 Hugging Face 下载模型、为 TensorRT 优化模型，以及在 Kubernetes 中配置自动扩缩容与负载均衡的完整步骤。本指南不涉及 Kubernetes 基础知识、集群到外部客户端的网络安全入口/出口（ingress/egress），也不涉及云厂商的 Kubernetes 接口或实现。

配置得当的自动扩缩容可以让基于 LLM 的服务根据当前负载自动分配和释放资源。在本教程中，随着某个 Triton Server 部署的客户端数量增长，服务器上的推理负载会随之增加，队列与计算时间之比（queue-to-compute ratio）最终会触发水平 Pod 自动扩缩器（Horizontal Pod Autoscaler，HPA）增加 Triton Server 实例数来处理请求，直到达到期望的比率。反之，客户端数量减少也会相应减少已部署的 Triton Server 实例数。

> 💡 **AI Infra 视角**：LLM 推理的自动扩缩容有两种主流思路：一种基于 QPS 等请求量指标（经典的 HPA 用法），另一种基于排队指标（如本指南的队列/计算比，或用 KEDA 监听队列深度）。对 LLM 来说，QPS 升高不等于服务过载——只要并发调度跟得上，吞吐还能继续涨；而队列积压才是"服务真的处理不过来了"的信号。所以生产环境里，用排队指标触发扩容通常比用 QPS 更可靠，这也是本指南选择 `queue_compute:ratio` 的原因。

我们将涵盖以下主题：

* [集群搭建（Cluster Setup）](#cluster-setup)
  * [核心集群服务（Core Cluster Services）](#core-cluster-services)
    * [Kubernetes Node Feature Discovery 服务](#kubernetes-node-feature-discovery-service)
    * [Kubernetes 的 NVIDIA Device Plugin](#nvidia-device-plugin-for-kubernetes)
    * [NVIDIA GPU Feature Discovery 服务](#nvidia-gpu-feature-discovery-service)
  * [指标采集服务（Metrics Collection Services）](#metrics-collection-services)
    * [创建监控命名空间（Monitoring Namespace）](#create-a-monitoring-namespace)
    * [Prometheus 服务（Prometheus Services）](#prometheus-services)
    * [NVIDIA Data Center GPU Manager (DCGM) Exporter](#nvidia-data-center-gpu-manager-dcgm-exporter)
    * [将 DCGM 和 Triton 指标接入 Prometheus](#connect-dcgm-and-triton-metrics-to-prometheus)
    * [Triton 指标 Prometheus Rule](#triton-metrics-prometheus-rule)
  * [创建 NFS](#nfs-creation)
* [Triton 准备（Triton Preparation）](#triton-preparation)
  * [Pod 初始化脚本](#pod-initialization-script)
  * [模型准备步骤](#model-preparation-steps)
  * [自定义容器镜像](#custom-container-image)
  * [Kubernetes 拉取凭据（Pull Secrets）](#kubernetes-pull-secrets)
* [Triton 部署（Triton Deployment）](#triton-deployment)
  * [部署单 GPU 模型](#deploying-single-gpu-models)
  * [部署单块 GPU 无法容纳的模型](#deploying-models-too-large-for-a-single-gpu)
  * [利用多种 GPU SKU](#utilizing-multiple-gpu-skus)
  * [在 Kubernetes 中监控 Triton](#monitoring-triton-in-kubernetes)
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

以下说明详细介绍了如何在 Kubernetes 集群中为 Triton Inference Server 配置水平 Pod 自动扩缩容（HPA）。

### 前置条件

本指南假定所有带 NVIDIA GPU 的节点都已具备以下配置：
- 节点标签 `nvidia.com/gpu=present`，用于更方便地识别带有 NVIDIA GPU 的节点。
- 节点污点 `nvidia.com/gpu=present:NoSchedule`，用于阻止非 GPU 的 Pod 被调度到 GPU 节点上。

> [!Tip]
> 如果使用 AKS、EKS 或 GKE 这类 Kubernetes 托管服务，配置节点时通常最好使用它们各自的控制界面，而不是直接用 `kubectl` 操作。

### 核心集群服务（Core Cluster Services）

所有节点正确打标签和设置污点后，按照以下步骤准备集群，以便采集和提供启用 Triton Server 水平自动扩缩容所需的指标。

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
    NAME                                          DESIRED  CURRENT  READY  UP-TO-DATE  AVAILABLE
    kube-proxy                                    6        6        6      6           6
    kube-system   node-feature-discovery-worker   1        1        1      1           1
    nvidia-device-plugin-daemonset                6        6        6      6           6
    ```

2.  如果列表中没有 `nvidia-device-plugin-daemonset`，请运行下面的命令安装该插件。
    安装后，它会让容器能够访问集群中的 GPU。

    更多信息参见
    [Github/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin/blob/main/README.md)。

    ```bash
    kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
    ```

> 💡 **AI Infra 视角**：Kubernetes 本身不认识 GPU，NVIDIA Device Plugin 正是让 GPU 成为可调度资源的关键组件。它作为 DaemonSet 在每个节点上运行，向 kubelet 上报节点的 GPU 数量，使调度器能根据 Pod 的 `nvidia.com/gpu` 资源申请做调度。注意 GPU 资源只有整卡粒度，不能像 CPU/内存那样按小单位切分——这也是 GPU 集群经常出现资源碎片、需要配组调度或配额管理的原因之一。

#### NVIDIA GPU Feature Discovery 服务

1.  如果你的集群已经安装了该服务，可以跳过这一步。

    要检查你的集群是否需要 NVIDIA GPU Feature Discovery 服务，请运行以下命令，并在输出中查找 `gpu-feature-discovery`。

    ```bash
    kubectl get daemonsets --all-namespaces
    ```

    示例输出：
    ```text
    kubectl get daemonsets --all-namespaces
    NAMESPACE     NAME                                  DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE
    kube-system   gpu-feature-discovery                 2         2         2       2            2
    kube-system   kube-proxy                            6         6         6       6            6
    kube-system   node-feature-discovery-worker         6         6         6       6            6
    kube-system   nvidia-device-plugin-daemonset        6         6         6       6            6
    ```

2.  如果列表中已有 `gpu-feature-discovery`，请跳过这一步和下一步。

    否则，请使用下面的 YAML 文件安装 GPU Feature Discovery 服务。

    > [nvidia_gpu-feature-discovery_daemonset.yaml](nvidia_gpu-feature-discovery_daemonset.yaml)

    上面的文件是从
    [GitHub/NVIDIA](https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.8.2/deployments/static/gpu-feature-discovery-daemonset.yaml)
    下载内容，并针对 Triton Server 自动扩缩容做了修改。

    ```bash
    curl https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.8.2/deployments/static/gpu-feature-discovery-daemonset.yaml \
      >  nvidia_gpu-feature-discovery_daemonset.yaml
    ```

3.  然后运行以下命令进行安装。

    ```bash
    kubectl apply -f ./nvidia_gpu-feature-discovery_daemonset.yaml
    ```

### 指标采集服务（Metrics Collection Services）

现在你的集群已经正常运行，并且可以为容器分配 GPU 资源了。
接下来，我们需要为 DCGM 和 Triton Server 设置指标采集。
指标服务为 Kubernetes 的 Horizontal Pod Autoscaler 提供利用率与可用性数据，这些数据随后可用于做出扩缩容决策。

> 💡 **AI Infra 视角**：本节从 Node Feature Discovery、Device Plugin、GPU Feature Discovery 到 DCGM Exporter、Prometheus、Prometheus Adapter，串起了"GPU 可调度 → 可监控 → 可弹性"的完整链路。其中 DCGM Exporter 提供 GPU 硬件级指标（利用率、显存、温度、功耗），Triton 自身暴露业务指标（队列时间、计算时间），两者互补：前者看"硬件是否吃饱"，后者看"服务是否够快"，缺一不可。

#### 创建监控命名空间（Monitoring Namespace）

在集群中创建 `monitoring` 命名空间，用于承载所有指标与监控服务。

1.  运行下面的命令创建命名空间。

    ```bash
    kubectl create namespace monitoring
    ```

#### Prometheus 服务（Prometheus Services）

我们需要一个服务来采集、存储、聚合集群及其部署服务的指标，并对外提供这些指标。
最简单的方法之一是利用
[Prometheus Metrics Server](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) 的功能。
按照下面的步骤，我们安装 Kubernetes 的 Prometheus Stack Helm chart，以便利用 Prometheus。

1.  将 Prometheus Community chart 仓库添加到本地缓存。

    ```bash
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts \
      && helm repo update
    ```

2.  运行下面的命令安装 Prometheus Kubernetes Stack Helm chart。

    ```bash
    helm install -n monitoring prometheus prometheus-community/kube-prometheus-stack \
      --set tolerations[0].key=nvidia.com/gpu \
      --set tolerations[0].operator=Exists \
      --set tolerations[0].effect=NoSchedule
    ```

    > [!Note]
    > 上面的命令设置了容忍（toleration）值，允许 Pod 被调度到带有匹配污点的节点上。
    > 参见本文档的[前置条件](#prerequisites)，了解本文档预期已应用到集群 GPU 节点上的污点。

#### NVIDIA Data Center GPU Manager (DCGM) Exporter

集群中管理 GPU 的最佳方案是
[NVIDIA DCGM](https://docs.nvidia.com/data-center-gpu-manager-dcgm)。
不过对于本示例，我们并不需要整个 DCGM 套件。
我们将按下面的步骤只安装 [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)，以便在集群中采集 GPU 指标。

1.  将 NVIDIA DCGM chart 仓库添加到本地缓存。

    ```bash
    helm repo add nvidia-dcgm https://nvidia.github.io/dcgm-exporter/helm-charts \
      && helm repo update
    ```

2.  使用下面的 YAML 文件安装 DCGM Exporter。

    > [nvidia_dcgm-exporter_values.yaml](nvidia_dcgm-exporter_values.yaml)

    上面的内容是用 `helm show values nvidia-dcgm/dcgm-exporter` 生成的，并针对 Triton Server 自动扩缩容做了修改。

4.  使用下面的命令安装 DCGM Exporter Helm chart。

    ```bash
    helm install -n monitoring dcgm-exporter nvidia-dcgm/dcgm-exporter --values nvidia_dcgm-exporter_values.yaml
    ```

#### 将 DCGM 和 Triton 指标接入 Prometheus

我们需要一种机制，把 Prometheus 服务器采集的指标导出出来，提供给 Kubernetes 的 Horizontal Pod Autoscaler 服务使用。
下面的步骤将安装一个 Prometheus Adapter，它创建一个自定义指标服务 API，HPA 服务可以用这个 API 从 Prometheus 读取指标。

1.  运行下面的命令安装 Prometheus Adapter Helm chart。

    ```bash
    helm install -n monitoring prometheus-adapter prometheus-community/prometheus-adapter \
      --set metricsRelistInterval=6s \
      --set customLabels.monitoring=prometheus-adapter \
      --set customLabels.release=prometheus \
      --set prometheus.url=http://prometheus-kube-prometheus-prometheus \
      --set additionalLabels.release=prometheus
    ```

2.  要验证 adapter 已正确安装和配置，请至少等待 60 秒，然后运行下面的命令。
    请注意，adapter 安装完成后，自定义指标还需要一段时间才会出现。

    ```bash
    kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1
    ```

    如果命令失败，请多等一会儿再重试。如果连续几分钟都失败，说明 adapter 配置有误，需要人工干预。

#### Triton 指标 Prometheus Rule

Prometheus rule 提供一种机制，用公式对 Prometheus 正在采集的数据进行计算，从而生成指标数据。
我们将创建一组针对 Triton Server 的 rule，生成对自动扩缩容有用的指标。

1.  使用下面的 YAML 内容创建一个名为 `triton-metrics_prometheus-rule.yaml` 的文件。

    > [triton-metrics_prometheus-rule.yaml](triton-metrics_prometheus-rule.yaml)

2.  运行下面的命令在集群中创建相应的 Prometheus Rule。_注意_，该 rule 会创建在当前上下文所在的命名空间中，通常是 `default`。
    如果想安装到其他命名空间，可以切换上下文，或者在命令中加上 `-n <desired_namespace>`。

    ```bash
    kubectl apply -f ./triton-metrics_prometheus-rule.yaml
    ```

在示例 Helm chart 的所有 values 文件中，水平 Pod 自动扩缩器都配置为使用上面 rule 提供的 `triton:queue_compute:ratio` 指标。
使用这个指标的好处是它与硬件和模型无关，因为它度量的是请求在推理队列中等待的时间与离开队列后完成处理所需时间之比。
这类指标允许对运行在不同硬件上的模型性能进行比较。

如果绝对响应时间对你更重要，那么 `triton:request_duration:average` 或 `triton:compute_duration:average` 这两个指标可能更符合需求。

> 💡 **AI Infra 视角**：为什么不用 GPU 利用率做扩容指标？因为 GPU 利用率是"事后"指标且噪声很大——模型加载、短请求、波峰波谷都会让利用率剧烈波动，利用率高也不代表延迟不可接受。而队列时间/计算时间之比直接度量"服务有多接近饱和"：比值大于 1 说明排队时间已超过处理时间，正是应该扩容的时刻。这个"饱和度指标"的思想（排队理论里的利用率 ρ）在 SRE 容量规划里是标准做法。

### 创建 NFS

要从 Hugging Face 下载模型并创建 TRT-LLM 模型，你需要一个 Pod 都能访问的 NFS。
我们不指定具体的 NFS 方案，本演示中使用 Amazon EFS（Elastic File System）。
为了让部署在不同节点上的多个 Pod 能够加载同一个模型的不同分片（shard），从而协同服务那些单块 GPU 无法承载的超大推理请求，我们需要一个共享存储位置。在 Kubernetes 中，这种共享存储位置称为持久卷（persistent volume）。持久卷可以同时映射挂载到任意数量的 Pod 中，Pod 内的进程可以像访问自己文件系统的一部分那样访问它。这里我们将使用 EFS 作为持久卷。

此外，我们还需要创建一个持久卷声明（persistent-volume claim，PVC），用它把持久卷分配给某个 Pod。
#### 1. 创建 IAM 角色

按照以下步骤为你的 EFS 文件系统创建 IAM 角色：https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html#efs-create-iam-resources。这个角色之后安装 EFS CSI Driver 时会用到。

#### 2. 安装 EFS CSI driver

在 AWS 控制台的 Amazon EKS 插件（Add-on）中安装 EFS CSI Driver：https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html#efs-install-driver。安装完成后，检查 EKS 控制台的 Add-ons 部分，应该能看到该驱动的 Status 显示为 `Active`。

#### 3. 创建 EFS 文件系统

按照以下步骤创建 EFS 文件系统：https://github.com/kubernetes-sigs/aws-efs-csi-driver/blob/master/docs/efs-create-filesystem.md。确保最后一步正确挂载子网，这直接关系到你的节点能否访问所创建的 EFS 文件系统。

#### 4. 测试 NFS

按照以下步骤检查你的 EFS 文件系统与节点是否工作正常：https://github.com/kubernetes-sigs/aws-efs-csi-driver/tree/master/examples/kubernetes/multiple_pods。这个测试会把 EFS 文件系统挂载到你所有可用的节点上，并向文件系统写入一个文本文件。

#### 5. 为创建的 EFS 文件系统创建 PVC

我们在这里提供了示例：[pvc_aws](./chart/pvc_aws/)。该文件夹包含三个文件：`pv_aws.yaml`、`claim_aws.yaml` 和 `storageclass_aws.yaml`。请务必修改 `pv_aws.yaml` 文件，把 `volumeHandle` 的值改成你自己的 EFS 文件系统 ID。

pv.yaml

```
apiVersion: v1
kind: PersistentVolume
metadata:
  name: efs-pv
spec:
  capacity:
    storage: 200Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: efs-sc
  csi:
    driver: efs.csi.aws.com
    volumeHandle: fs-0cf1f987d6f5af59c # Change to your own ID
```

claim.yaml

```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: efs-claim
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: efs-sc
  resources:
    requests:
      storage: 200Gi
```

storageclass.yaml

```
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
```

运行以下命令进行部署：

```
kubectl apply -f pvc/
```

#### 6. 修改 deployment.yaml 中的 claimName，指向你的 chart/templates/deployment.yaml

```
persistentVolumeClaim:
  claimName: nfs-claim-autoscaling-2 (Edit your claimName)
```

> 💡 **AI Infra 视角**：自动扩缩容会反复创建新 Pod，如果每个新 Pod 都要重新下载模型、重新构建 TRT-LLM 引擎（动辄几十分钟），扩容就失去了意义。把模型仓库放到共享存储（NFS/EFS）上，让新 Pod 直接挂载现成的引擎文件，扩容才能做到分钟级乃至秒级。这也是推理平台的标准做法：模型资产与计算资源解耦，模型放存储层、GPU 放计算层。

## Triton 准备（Triton Preparation）

### Pod 初始化脚本

1.  创建一个名为 `server.py` 的 Python 文件，内容见下面链接。

    > [server.py](containers/server.py)

    这个方案还可以进一步改进：增加一个集群内所有节点共享的网络存储位置，用它全局缓存每个模型/GPU 组合的 plan 和 engine 文件。
    之后，新节点上的 Pod 启动时，如果 GPU 型号相同，就可以直接下载已生成的文件，而不用在本地重新生成。
    与在本地生成文件相比，这可以节省大量时间（下载文件通常比生成快得多，至少省几秒钟）。

### 模型准备步骤：

要在计算节点内构建 TRT-LLM 引擎并配置 Triton 模型仓库，请按以下步骤操作：

1.  修改 `setup_ssh_nfs.yaml` 文件

    我们使用 `setup_ssh_nfs.yaml` 文件（其执行"sleep infinity"）在计算节点内配置 ssh 访问并挂载 EFS。

    需要调整以下值：

    - `image`：修改镜像标签。默认是 24.08，支持 TRT-LLM v0.12.0
    - `nvidia.com/gpu`：设为集群中每个节点的 GPU 数量，在 limits 和 requests 两处都要修改
    - `claimName`：设为你的 EFS pvc 名称

2.  SSH 进入计算节点并构建 TRT-LLM 引擎

    部署 Pod：

    ```
    cd multinode_helm_chart/
    kubectl apply -f setup_ssh_nfs.yaml
    kubectl exec -it setup-ssh-nfs -- bash
    ```

    克隆 Triton TRT-LLM 后端仓库：

    ```
    cd <EFS_mount_path>
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.12.0
    cd tensorrtllm_backend
    git lfs install
    git submodule update --init --recursive
    ```

    构建一个张量并行度=1、流水线并行度=1 的 Llama3-8B 引擎
    ```
    cd tensorrtllm_backend/tensorrt_llm/examples/llama

    pip install -U "huggingface_hub[cli]"
    huggingface-cli login
    huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./Meta-Llama-3-8B --local-dir-use-symlinks False

    python3 convert_checkpoint.py --model_dir ./Meta-Llama-3-8B \
                                --output_dir ./converted_checkpoint \
                                --dtype bfloat16 \
                                --tp_size 1 \
                                --pp_size 1 \
                                --load_by_shard \
                                --workers 1

    trtllm-build --checkpoint_dir ./converted_checkpoint \
                --output_dir ./output_engines \
                --max_num_tokens 4096 \
                --max_input_len 65536 \
                --max_seq_len 131072 \
                --max_batch_size 8 \
                --use_paged_context_fmha enable \
                --workers 1
    ```

3.  准备 Triton 模型仓库

    ```
    cd <EFS_MOUNT_PATH>/tensorrtllm_backend
    mkdir triton_model_repo

    cp -r all_models/inflight_batcher_llm/ensemble triton_model_repo/
    cp -r all_models/inflight_batcher_llm/preprocessing triton_model_repo/
    cp -r all_models/inflight_batcher_llm/postprocessing triton_model_repo/
    cp -r all_models/inflight_batcher_llm/tensorrt_llm triton_model_repo/

    python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt tokenizer_dir:<PATH_TO_TOKENIZER>,tokenizer_type:llama,triton_max_batch_size:8,preprocessing_instance_count:1
    python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:8,decoupled_mode:True,max_beam_width:1,engine_dir:<PATH_TO_ENGINES>,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:0
    python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt tokenizer_dir:<PATH_TO_TOKENIZER>,tokenizer_type:llama,triton_max_batch_size:8,postprocessing_instance_count:1
    python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt triton_max_batch_size:8
    ```

    > [!Note]
    > 请务必把上面示例中的 `<PATH_TO_TOKENIZER>` 和 `<PATH_TO_ENGINES>` 替换成正确的值。请记住，tokenizer、TRT-LLM 引擎和 Triton 模型仓库必须放在节点之间共享的文件存储中，它们是 Triton 启动模型所必需的。例如，如果使用 AWS EFS，`<PATH_TO_TOKENIZER>` 和 `<PATH_TO_ENGINES>` 的值应相对于实际的 EFS 挂载路径，这个路径由你的持久卷声明和 chart/templates/deployment.yaml 中的挂载路径决定。请确保你的节点能够访问这些文件。

4.  删除 Pod

    ```
    exit
    kubectl delete -f setup_ssh_nfs.yaml
    ```

#### 自定义容器镜像

1.  使用下面的文件，我们在下一步创建一个自定义容器镜像。

    > [triton_trt-llm.containerfile](containers/triton_trt-llm.containerfile)

2.  运行下面的命令创建一个自定义的 Triton Inference Server 镜像，包含生成 TensorRT-LLM plan 和 engine 文件所需的全部工具。本例使用标签 `24.08`，与基础镜像的 `24.08-trtllm-python-py3` 日期部分保持一致。

    ```bash
    docker build \
      --file ./triton_trt-llm.containerfile \
      --rm \
      --tag triton_trt-llm:24.08 \
      .
    ```

3.  将容器镜像上传到集群可访问的仓库。

    要让 Kubernetes 集群能够下载我们的新容器镜像，必须把它推送到集群节点可以访问的容器镜像仓库。
    本例使用虚构的 `nvcr.io/example` 仓库做演示。
    你需要确定自己有哪些可写权限的仓库，并且你的集群也能访问这些仓库。

    1. 首先，像下面这样用仓库名重新标记容器镜像。

        ```bash
        docker tag \
          triton_trt-llm:24.08 \
          nvcr.io/example/triton_trt-llm:24.08
        ```

    2. 接下来，把容器镜像上传到你的仓库。

        ```bash
        docker push nvcr.io/example/triton_trt-llm:24.08
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

### 部署单 GPU 模型

按照下面的步骤，部署一个能装进单块 GPU 的模型到 Triton Server 上，非常简单直接。

1.  创建一个包含必需值的自定义 values 文件：

    * 容器镜像名称。
    * 模型名称。
    * 支持/可用的 GPU。
    * 镜像拉取凭据（如有必要）。

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
    kubectl get deployments,pods,hpa,services,podmonitors --selector='app=<installation_name>'
    ```

    > [!Important]
    > 请务必把上面示例中的 `<installation_name>` 替换成正确的值。

    输出应该类似下面这样（假设安装名称为 "llama-3"）：

    ```text
    NAME                      READY   UP-TO-DATE   AVAILABLE
    deployment.apps/llama-3   0/1     1            0

    NAME                          READY   STATUS    RESTARTS
    pod/llama-3-7989ffd8d-ck62t   0/1     Pending   0

    NAME                                          REFERENCE            TARGETS   MINPODS   MAXPODS   REPLICAS
    horizontalpodautoscaler.autoscaling/llama-3   Deployment/llama-3   0/1       1         8         1

    NAME              TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)
    service/llama-3   ClusterIP   10.100.23.237   <none>        8000/TCP,8001/TCP,8002/TCP

    NAME
    podmonitor.monitoring.coreos.com/llama-3
    ```

    HPA 的 `TARGETS` 可能显示为 `<unknown>/1`。
    这不一定是有问题的。很可能是还没有客户端应用向 Triton Server 发送推理查询。
    没有推理查询，就不会产生指标，因此 HPA 控制器把指标当前值报告为 `<unknown>`。
    一旦客户端开始向 Triton Server 发送推理查询，这个问题就会自行解决。

4.  卸载 Chart

    卸载 Helm chart 很简单，运行下面的命令即可。
    这在尝试各种选项和配置时非常有用。

    ```bash
    helm uninstall <installation_name>
    ```

### 部署单块 GPU 无法容纳的模型

考虑到某些 AI 模型的内存需求，单块设备无法承载它们。
Triton 和 TensorRT-LLM 提供了一种机制，让多个 GPU 设备协同工作来承载大模型。
提供的示例 Helm [chart](./chart/) 就提供了利用这一能力的机制。

要启用该功能，请把 `model.tensorrtLlm.parallelism.tensor` 的值改为大于 1 的整数。
为模型配置张量并行（tensor parallelism）后，TensorRT-LLM 运行时会高效地合并多块 GPU 的内存，从而承载单块 GPU 装不下的模型。

同样，修改 `model.tensorrtLlm.parallelism.pipeline` 的值可以启用流水线并行（pipeline parallelism）。
流水线并行用于合并多块 GPU 的计算能力，并行处理推理请求。

承载模型所需的 GPU 数量等于 `.tensor` 和 `.pipeline` 值的乘积。
需要特别注意的是，用于承载模型的 GPU 必须位于同一节点上。

> [!Note]
> 合并不同节点上的 GPU 不在本指南的讨论范围内（跨节点的方案请参考多节点分布式模型教程）。

> 💡 **AI Infra 视角**：张量并行（TP）把模型的一层权重切分到多块 GPU 上，每步计算都伴随 AllReduce 通信，所以依赖节点内的高速互连（NVLink/NVSwitch）；流水线并行（PP）按层分段，通信量小但有空转气泡。两者组合（如 TP=4、PP=2）是 LLM 部署的常见形态，GPU 总数 = TP × PP。K8s 里 Pod 可以声明多块 GPU，调度器会把它们放到同一个节点上——因为跨节点通信无法满足 TP 的性能要求。

### 利用多种 GPU SKU

鉴于某些 GPU SKU 相对稀缺，服务需要运行在混合 GPU 硬件上的情况并不少见。
例如，集群中基于 NVIDIA Hopper 架构的设备数量可能不足以满足负载需求，但还有配备 NVIDIA Ampere 架构设备的空闲节点。
这种情况下，合理的做法是按照上面的[步骤](#deploying-single-gpu-models)为同一个模型创建多个 deployment，并把它们全部放在同一个 Kubernetes Service 后面做负载均衡。
这样两种 SKU 的设备都能独立自动扩缩容，共同为服务提供算力。

要实现这一点，我们可以修改 chart，让它在部署时不创建 Service，而是使用共享 Service 指定的选择器标签。
下面的示例中，我们假定该 Service 已经创建好，其选择器设置为 `model=llama-3-8b`。

```bash
helm install llama-3-8b-a100 ./chart/. \
  --values ./chart/values.yaml \
  --values ./chart/llama-3-8b \
  --set 'triton.image.name=<custom_image_name>' \
  --set 'gpu[0]=NVIDIA-A100-SXM4-40GB' \
  --set 'kubernetes.labels[0].model=llama-3-8b' \
  --set 'kubernetes.noService=true'

helm install llama-3-8b-h100 ./chart/. \
  --values ./chart/values.yaml \
  --values ./chart/llama-3-8b \
  --set 'triton.image.name=<custom_image_name>' \
  --set 'gpu[0]=NVIDIA-H100-SXM5-80GB' \
  --set 'kubernetes.labels[0].model=llama-3-8b' \
  --set 'kubernetes.noService=true'
```

结果是集群中会出现两个 deployment，它们都属于这个 Service 的负载均衡池。

```bash
kubectl get deployments --selector='model=llama-3-8b'
NAME                    READY   UP-TO-DATE   AVAILABLE
llama-3-8b-a100         1/1     1            1
llama-3-8b-h100         1/1     1            1
```

> 💡 **AI Infra 视角**：多 SKU 混部是降本的常用手段：把部分流量调度到便宜/空闲的 A100 上，用 H100 承接高优先级流量，同一个 Service 后面挂两个 deployment，K8s 的 kube-proxy 按 round-robin 分发。两个 deployment 的 HPA 独立扩缩，正好利用各自的资源池。代价是两套实例延迟/吞吐特性不同，需要考虑 SLO 分层和容量规划。生产上还可以更进一步，用流量镜像或基于延迟的路由做精细化分流。

### 在 Kubernetes 中监控 Triton

可以使用本文档 [Prometheus 服务（Prometheus Services）](#prometheus-services) 一节安装的 Prometheus 软件来监控 Kubernetes 中的 Triton。
安装的软件包含一个 Grafana 仪表盘服务器。
要连接 Grafana 服务器，我们首先需要从本地工作站建立一个到集群的网络隧道。

1.  运行下面的命令，从本地机器建立到 Kubernetes 集群的网络隧道。

    ```bash
    kubectl port-forward -n monitoring svc/prometheus-grafana 8080:80
    ```

    这会在本地机器的 `8080` 端口与集群内 Grafana 服务器的 `80` 端口之间建立隧道。
    成功后，你应该会看到类似下面的输出。

    ```bash
    Forwarding from 127.0.0.1:8080 -> 3000
    Forwarding from [::1]:8080 -> 3000
    ```

2.  打开浏览器，在地址栏输入 `http://http://127.0.0.1:8080/`。

3.  第一次访问时，需要登录 Grafana。
    使用下面的用户名和密码完成登录。

    * 用户名：`admin`
    * 密码：`prom-operator`

    > [!Tip]
    > 这是 Grafana 作为 Prometheus Helm chart 一部分安装时的默认用户名和密码。

4.  我们要做的第一件事是创建一个新的自定义仪表盘。
    点击用户界面右上角的 `+` 图标，从下拉菜单中选择 `New dashboard`。

    ![可视化 "new dashboard" 界面](./images/grafana_new-dashboard.png)

5.  Grafana 会询问你想如何创建新仪表盘。
    选择 `Import dashboard` 选项。

    ![可视化 "new dashboard" 界面](./images/grafana_import-dashboard.png)

6.  把提供的 [grafana_inference-metrics_dashboard.json](./grafana_inference-metrics_dashboard.json) 文件内容复制粘贴到名为 `Import via dashboard JSON model` 的文本框中，或者用界面中的 `Upload dashboard JSON file` 工具上传该文件。

7.  创建新仪表盘后，你应该会看到类似下图的界面。

    ![按照上述说明创建的 Grafana 仪表盘示例。](./images/grafana-dashboard.png)

仪表盘配置好后，你就能可视化集群的当前状态。
这些可视化图表有助于理解我们为什么选择用队列/计算比（queue:compute ratio）而不是 GPU 利用率作为控制水平 Pod 自动扩缩器的指标。

| GPU 利用率                                                      | 队列/计算比                                                       |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------  |
| ![GPU 利用率图表示例](./images/graph_gpu-utilization.png) | ![队列/计算比图表示例](./images/graph_queue-compute-ratio.png) |

上面两张图是同一时间段的数据。
对比二者可以清楚看到，比率图能更清晰地表明何时需要增加资源来满足当前推理需求，而 GPU 利用率图的噪声太大，无法为水平 Pod 自动扩缩器提供清晰的信号。

## 本指南的开发过程（Developing this Guide）

在编写本指南的过程中，我遇到了几个必须先解决的问题，然后才能写出一份有用的指南。
本节将概述我在开发过程中遇到的问题以及解决方法。

> _本文档是使用 Amazon EKS 提供的 Kubernetes 集群开发的。_
> _本地机房或 Azure AKS、GCloud GKE 等其他云厂商提供的集群可能需要对本文档做相应修改。_

### 指标配置既是科学也是艺术

在编写本指南期间，我花了大量时间弄清楚采集所有必要且有用指标所需的每一个变量、设置和配置。
其中大部分精力花在弄清 Kubernetes HPA 控制器的细节以及它如何消费指标上。

起初，我无法让 HPA 控制器识别我想用来控制 Pod 扩缩容的自定义指标。
最终我发现，安装 [Kubernetes 的 Prometheus Stack](#prometheus-services) 时，v2 版 HPA 控制器已被自动配置为使用 Prometheus 提供的 `custom.metrics.k8s.io/v1beta1` 端点。

运行下面的命令，可以获取 `custom.metrics.k8s.io/v1beta1` 端点提供的指标集合。

```bash
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1
```

上面会返回一个 JSON 数据块，可以在你喜欢的 IDE 中检查。
我推荐 VSCode，因为它对 JavaScript 和 JSON 支持很好，不过用你顺手的工具就好。

可以用类似下面的命令从该端点查询当前指标值。

```bash
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pod/*/triton:queue_compute:ratio
```

上面的命令请求 `default` 命名空间中所有 Pod 的 `triton:queue_compute:ratio` 指标。
这几乎正是 Kubernetes v2 HPA 控制器查询它做扩缩容决策所需指标的方式。
知道这一点之后，我就能在 Prometheus 和 Prometheus Adapter 安装的配置值、以及我们创建的 Prometheus Rule 中反复试验，直到一切"正好工作"。

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

#### Kubernetes 的 Prometheus Stack

为集群提供指标采集和聚合服务。
虽然还有其他工具也能提供类似服务，但我们发现 Kubernetes 的 Prometheus Stack 最容易安装和配置。
此外，它自动附带基于 Grafana 的用户界面，让集群当前健康状况的可视化更容易搭建。

我们最初的工作是基于另一个指标服务，但我们发现从 Triton Server 采集指标、以及用自定义指标驱动水平 Pod 自动扩缩容，都过于困难和混乱。

#### Kubernetes 的 Prometheus Adapter

提供从非标准指标源（如 Triton Server）采集指标的能力，这是利用本文档所述自定义指标时的必要前提。

##### 为什么 Prometheus Adapter 的 values 文件是自定义的？

我为 Prometheus Adapter 创建自定义 values 文件的原因，与为 DCGM Exporter 创建自定义 values 文件的原因非常相似。
污点与容忍、针对指标采集优化的值，以及必须提供已部署 Prometheus 服务器的正确 URL。

### 为什么 Chart 运行 Python 脚本而不是直接运行 Triton Server？

有两个原因：

1.  为了从 Hugging Face 获取模型、转换并优化成 TensorRT-LLM 格式、再在宿主机上缓存，我认为使用 [Pod 初始化容器](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/) 是最直接了当的方案。

    为了充分利用初始化容器，我选择了自定义的 [server.py](./containers/server.py)。

2.  多 GPU 部署需要相当专用的命令行才能运行，而我不想用 Helm chart 脚本去生成它。
    利用自定义 Python 脚本是合理且最简单的方案。

#### 为什么 Python 代码写成那样？

因为我不是 Python 开发者，但我正在学！
我的背景是 C/C++，有大量 shell 脚本经验。

### `client/` 文件夹是干什么的？

我决定把验证本指南时使用的工具也一并放进来，`client/` 文件夹中的部署定义就是其中关键的一部分。
如果你想的话，也可以自己使用它们。
只需要运行 `kubectl apply -f ./clients/llama-3-8b.yaml`（以 `llama-3-8b` 为例）创建 deployment，然后运行 `kubectl scale deployment/llama-3-8b --replicas=<number_of_desired_clients>` 调整副本数即可。

随着为某个 Triton Server deployment 生成推理请求的客户端数量增加，服务器上的负载会上升，队列与计算时间之比最终会触发水平 Pod 自动扩缩器增加 Triton Server 实例数来处理请求，直到达到期望的比率。

减少客户端数量会产生相反的效果，减少已部署的 Triton Server 实例数。

> [!Note]
> 在集群中创建客户端 deployment 之前，务必先使用 `containers/client.containerfile` 构建客户端容器镜像。
> 与构建 `containers/triton_trt-llm.containerfile` 时一样，镜像需要托管在集群机器能够下载的地方。

### 为什么本指南不包含负载均衡器（Load Balancer）说明？

我们实验过专门的负载均衡器——它们可以利用 Pod 指标决定把新请求发给哪个 Triton Server 实例——结果显示相比 Kubernetes 网络层通过 [kube-proxy](https://kubernetes.io/docs/reference/command-line-tools-reference/kube-proxy/) 提供的 "round robin" 机制，提升充其量只是"聊胜于无"。
反正 kube-proxy 是集群中每次网络操作都必需的，利用现有方案是更优的选择，因为它避免了在缺乏合理价值的情况下增加更多复杂度。

你的环境中的结果很可能不同。
我鼓励你实验一下专门的负载均衡器，为你的工作负载找到最佳方案。

> 💡 **AI Infra 视角**：这节的结论在 AI 推理场景里很有代表性：LLM 服务通常是无状态的（模型常驻显存，请求间无状态），因此 round-robin 负载均衡已经足够；真正的瓶颈不在分发策略，而在队列管理与后端饱和度感知。但如果你使用 KV cache 等有状态优化，或者实例间延迟差异大，基于指标感知的负载均衡（如 Envoy 的 least_request、P2C）才会带来实质收益。

---

本文档涉及的软件版本：

* Triton Inference Server v2.45.0 (24.08-trtllm-python-py3)
* TensorRT-LLM v0.9.0
* Kubernetes 的 NVIDIA Device Plugin v0.15.0
* Kubernetes 的 NVIDIA GPU Discovery 服务 v0.8.2
* NVIDIA DCGM Exporter v3.3.5
* Kubernetes Node Discovery 服务 v0.15.4
* Kubernetes 的 Prometheus Stack v58.7.2
* Kubernetes 的 Prometheus Adapter v4.10.0

---

作者：J Wyman，系统软件架构师，AI 与分布式系统

Copyright &copy; 2024, NVIDIA CORPORATION. All rights reserved.
