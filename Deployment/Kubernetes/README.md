# Kubernetes 部署 Triton Server 指南

> 💡 **AI Infra 视角**：生产环境部署推理服务几乎都会选择 Kubernetes，核心原因有三：一是弹性，Pod 可以随负载自动扩缩容，也能让集群自动加节点；二是资源隔离，通过 namespace、污点（taint）与容忍（toleration）把 GPU 节点与普通节点隔离开；三是可移植性，同一套 Helm chart 可以在 EKS、AKS、GKE 甚至自建集群上复用。下面的两个教程分别覆盖"单节点多 GPU 的自动扩缩容"与"跨节点多 GPU 的分布式推理"两种典型场景。

* [TensorRT-LLM 生成式 AI 自动扩缩容与负载均衡](./TensorRT-LLM_Autoscaling_and_Load_Balancing/README.md)
* [基于 Triton Server 和 TensorRT-LLM 的多节点生成式 AI](./TensorRT-LLM_Multi-Node_Distributed_Models/README.md)
