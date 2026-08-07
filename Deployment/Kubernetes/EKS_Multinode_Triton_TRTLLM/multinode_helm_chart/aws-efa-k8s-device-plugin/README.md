# AWS EFA Kubernetes Device Plugin

本 chart 用于安装 AWS EFA Kubernetes Device Plugin 的 DaemonSet

## 前置条件
- Helm v3

## 安装 Chart

首先将 EKS 仓库添加到 Helm：

```shell
helm repo add eks https://aws.github.io/eks-charts
```

以 release 名称 `efa` 在 `kube-system` 命名空间中安装该 chart，并使用默认配置：

```shell
helm install efa ./aws-efa-k8s-device-plugin -n kube-system
```

# 配置

参数 | 说明 | 默认值
--- | --- | ---
`image.repository` | EFA 镜像仓库 | `602401143452.dkr.ecr.us-west-2.amazonaws.com/eks/aws-efa-k8s-device-plugin`
`image.tag` | EFA 镜像标签 | `v0.5.3`
`securityContext.allowPrivilegeEscalation` | 控制进程是否可以获得比其父进程更高的权限 | `false`
`securityContext` | EFA 插件的安全上下文 | `capabilities: drop: ["ALL"] runAsNonRoot: false`
`supportedInstanceLabels.keys` | 用作实例类型判定的 Kubernetes 标签键 | `nodes.kubernetes.io/instance-type`
`supportedInstanceLabels.values` | 当前支持 EFA 设备的实例列表 | `参见 https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types`
`resources` | Pod 中容器的资源 | `requests.cpu: 10m requests.memory: 20Mi`
`nodeSelector` | Pod 调度到节点上所需的节点标签 | `{}`
`tolerations` | 可选的部署容忍（toleration）配置 | `[]`
`additionalPodAnnotations` | 除默认注解外额外附加的 Pod 注解 | `{}`
`additionalPodLabels` | 除默认标签外额外附加的 Pod 标签 | `{}`
`nameOverride` | 覆盖 chart 的名称 | `""`
`fullnameOverride` | 覆盖 chart 的完整名称 | `""`
`imagePullSecrets` | Docker 镜像仓库的拉取凭据 | `[]`

> 💡 **AI Infra 视角**：EFA 设备插件和 NVIDIA Device Plugin 扮演的角色一样，都是把"非 CPU/内存"的硬件资源暴露给 Kubernetes 调度器。它在每个 GPU 节点上以 DaemonSet 形式运行，把 `vpc.amazonaws.com/efa` 作为一种可调度资源注册到节点上，Pod 只要在 resources 里声明 `vpc.amazonaws.com/efa: 1` 就能申请到 EFA 网卡——这也是理解"设备插件（device plugin）"机制的最佳范例：任何加速硬件（GPU、网卡、FPGA）都可以通过同一套扩展机制接入 K8s 调度体系。
