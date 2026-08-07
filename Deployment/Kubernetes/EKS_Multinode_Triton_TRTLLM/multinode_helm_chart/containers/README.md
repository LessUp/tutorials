<!---
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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


# 容器镜像构建

本目录下的文件用于构建多节点 Triton + TRT-LLM EKS 部署所需的定制容器镜像，其中包含 EFA 组件的安装。

运行以下命令构建容器镜像：

```bash
docker build --file ./triton_trt_llm.containerfile --tag <image_name_here> .
```

> 💡 **AI Infra 视角**：这个镜像是在 NGC 官方 Triton TRT-LLM 镜像之上叠加 EFA 库和 kubessh 工具链做成的"胖镜像"。多节点 MPI 分布式推理对节点间的通信软件栈要求苛刻（需要与内核模块匹配的 libfabric/EFA 库版本），所以通常不会在运行时临时安装，而是直接打进镜像，保证每个节点上的环境完全一致，这也是镜像分层构建的典型实践。
