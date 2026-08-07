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

本目录下的文件用于构建 Triton Server 容器镜像。

运行以下命令构建 Triton Server 容器镜像：

```bash
docker build --file ./triton_trt-llm.containerfile --tag <image_name_here> .
```

> 💡 **AI Infra 视角**：镜像里除了 Triton Server 之外还打包了 triton CLI 等模型转换工具链，让"转换模型 + 启动服务"可以在同一个镜像内完成。生产环境中这是很常见的取舍：要么维护一个"转换专用镜像 + 服务镜像"的双镜像体系，要么像这里一样用一个全能镜像降低运维复杂度——代价是镜像体积变大、拉取变慢，需要根据实际发布频率权衡。
