<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 为 RHEL / manylinux 构建 Triton Inference Server

> [!WARNING]
> **这是社区示例——并非官方支持。** Triton 的
> `build.py --target-platform=rhel` 路径目前是实验性的（RHEL 不是官方支持的目标平台），
> 本教程重建的基础镜像与 NVIDIA 内部用于产出官方 `manylinux`
> 构件的镜像是**等价**的，但并不**完全相同**。

NVIDIA 会发布预编译的 `manylinux`（兼容 RHEL 8）Triton Inference Server 构件。要用
`build.py --target-platform=rhel` 自行复现，需要一份没有公开发布的
CUDA/cuDNN/TensorRT 基础镜像，所以本教程从公开来源重建一个等价镜像——基于 Rocky Linux 8
的公共 NVIDIA CUDA 镜像，加上来自公共 CUDA 仓库的 TensorRT——并完整走一遍
RHEL/manylinux Triton 服务器的构建与运行流程。

> 💡 **AI Infra 视角**：RHEL 是企业生产环境最常见的 Linux 发行版之一——长期支持、安全补丁和厂商背书让它成为金融、制造等行业的默认选择。但 Triton 官方预编译构件主要面向 Ubuntu，很多企业因此卡在"官方支持矩阵之外"。这篇教程的价值就在于：把 RHEL 生态下的构建能力掌握在自己手里，而不依赖 NVIDIA 是否官方支持。

本教程结束时，我们将产出以下成果：

1. **manylinux 构件。** `build/install/` 中存放 `tritonserver` / `tritonfrontend` 的
   wheel（标签形如 `…-cp312-cp312-manylinux_2_XX_x86_64.whl`），以及我们构建的后端树（如 `onnxruntime`、`pytorch`、`python`）——全部来自 100% 公开的输入。
2. **一个可用且可验证的 manylinux 容器。** 构建出的镜像能真实跑推理，wheel 和运行时都经验证符合 `manylinux` 规范（glibc 2.28 / EL8），因此可以在 RHEL 8 及其衍生版（Rocky、AlmaLinux 等）上运行。在 RHEL 9 上也能跑，但服务器二进制需要 OpenSSL 1.1（`libssl.so.1.1`）（见[与官方构件的差异](#与官方构件的已知差异)）。

下面命令的目标版本为 **Triton 2.69.0 / NGC 26.05**（CUDA 13.2.1、TensorRT 10.16.1.11、PyTorch 2.13.0、Python 3.12）。

> [!IMPORTANT]
> **想针对其他 Triton 版本？** 先在 NVIDIA 的
> [Framework Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
> 和 [Triton Inference Server 发布说明](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
> 中查好对应的 CUDA、TensorRT、PyTorch 和 Python 版本，然后把以下**所有**参数同步更新：
>
> - `--build-arg BASE_IMAGE=...`（第 1 步）
> - `--build-arg TENSORRT_VERSION=...`（第 1 步）
> - `--build-arg TORCH_VERSION=...`——[`Dockerfile.pytorch.rhel`](Dockerfile.pytorch.rhel)
>   **和** [`Dockerfile.pytorch-runtime.rhel`](Dockerfile.pytorch-runtime.rhel) **两处都要改**（第 2 步和第 4 步）
> - `--build-arg TORCH_INDEX_URL=...`——如果 CUDA 通道变了（例如 `cu132` → `cu133`）
> - `build.py` 调用里的 `--version`、`--container-version`、`--upstream-container-version` 以及每个 `--repo-tag`/`--backend=X:tag`（第 3 步）
>
> 尤其是 torch，两个 Dockerfile 里的版本必须**一致**——版本不匹配会在服务器加载时
> 触发 ABI 崩溃，而不是在构建时报错。

## 前置条件

- **一台 Linux x86-64 主机，Docker daemon 可用。** 本教程和标准 Triton 构建一样使用 `build.py`；Triton 官方支持的构建平台是
  [Ubuntu 22.04, x86-64](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md)。
  由于构建在容器中进行，其他装了 Docker 的 x86-64 Linux 主机理论上也能工作。
  **构建不需要 GPU**——CUDA 库来自基础镜像；只有跑 GPU 推理时才需要 GPU。
- **磁盘空间和构建时间取决于构建哪些后端。** 仅从源码构建 `onnxruntime`
  后端就需要约 2 小时和几十 GB 空间；最小化构建则快得多。
- 能访问公共的 **NVIDIA** 软件包仓库、**PyPI** 和 **GitHub**
  （`build.py` 会克隆后端源码）。
- [`Dockerfile.base.rhel`](Dockerfile.base.rhel) —— **必需**；任何
  `--target-platform=rhel` 构建都需要这个基础镜像。
- [`Dockerfile.pytorch.rhel`](Dockerfile.pytorch.rhel) —— **可选**；仅当构建
  `pytorch` 后端时需要（第 2 步）。
- [`Dockerfile.pytorch-runtime.rhel`](Dockerfile.pytorch-runtime.rhel) —— **可选**；
  补全构建镜像，让 `pytorch` 后端能对外服务（第 4 步）。

> [!NOTE]
> **选择后端。** 第 3 步的命令里去掉某个 `--backend=` 参数即可跳过对应后端；
> 去掉 `onnxruntime` 还能省下约 2 小时的源码构建。PyTorch 镜像（第 2 步）仅为 `pytorch` 后端所需。
>
> TensorRT 是个例外：在 x86-64 上为 `rhel` 构建 GPU 版 `onnxruntime` 时，会自动启用 ONNX Runtime 的
> TensorRT provider，所以即使不选 `tensorrt` 后端也会把它带进来。想要不含 TensorRT 的构建，
> 要么同时去掉 `onnxruntime` 和 `tensorrt`，要么保留 `onnxruntime` 并用
> `--override-backend-cmake-arg onnxruntime:TRITON_ENABLE_ONNXRUNTIME_TENSORRT=OFF` 禁用该 provider；
> 同时把 TensorRT 步骤从 `Dockerfile.base.rhel` 中去掉。

## 第 1 步：构建公共基础镜像

在 `rhel` 路径下，`build.py` 只负责安装 DCGM（NVIDIA 数据中心 GPU 管理器），
CUDA/cuDNN/TensorRT 技术栈和一批操作系统 `-devel` 包需要基础镜像中已经备好。
[`Dockerfile.base.rhel`](Dockerfile.base.rhel) 从公开来源重建了这一切：以 NVIDIA 官方
`nvidia/cuda:*-cudnn-devel-rockylinux8` 镜像为起点（CUDA + cuDNN，基于 Rocky Linux 8 = RHEL 8 /
glibc 2.28），启用 **EPEL + PowerTools**，从公共 `cuda-rhel8` 仓库安装 **TensorRT**，
并补齐 `build.py` 所需的编译器工具链、Python 头文件和 wheel 工具链。
（PyTorch 额外的 CUDA 运行时库——cuSPARSELt、NCCL、nvshmem——*不*在这里添加；
它们随 torch wheel 一起分发，在第 4 步的补全镜像中接线，基础镜像因此保持通用。）

构建该镜像，运行下面的命令，把 CUDA 镜像和 TensorRT 固定到你的发布版本：

```bash
docker build -f Dockerfile.base.rhel \
  --build-arg BASE_IMAGE=nvidia/cuda:13.2.1-cudnn-devel-rockylinux8 \
  --build-arg TENSORRT_VERSION=10.16.1.11-1.cuda13.2 \
  -t triton-manylinux-base:example .
```

> 💡 **AI Infra 视角**：官方已经发布了预编译 wheel，为什么还要从源码构建？对企业 AI Infra 来说，可复现性比"省事"更重要：自建构建流水线意味着可以固定每个依赖版本、把产物纳入自己的供应链，还能覆盖官方支持矩阵之外的平台。容器在这里是构建隔离的关键——所有编译都在镜像内进行，宿主机只需要一个 Docker daemon，构建环境不会与生产环境互相污染。

## 第 2 步（可选）：构建 PyTorch 后端镜像

仅构建 `pytorch` 后端时需要。
[`Dockerfile.pytorch.rhel`](Dockerfile.pytorch.rhel) 将一个公共 `torch` wheel 安装到
`manylinux_2_28` 镜像中，这样 PyTorch 后端就能提取出可在 EL8 的 glibc 2.28 上运行的 libtorch。
注意，默认基于 Ubuntu 的 `libtorch` 是针对更新的 glibc 构建的，在那里无法加载。

```bash
docker build -f Dockerfile.pytorch.rhel \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
  -t triton-manylinux-pytorch:example .
```

注意：`build.py` 消费 `--image=pytorch` 时无条件执行 `docker pull`（即使在
`--no-container-pull` 下也一样），所以该镜像必须能从某个 registry 拉取。把它推送到一个临时本地 registry：

```bash
docker run -d -p 5000:5000 --name registry registry:2
docker tag  triton-manylinux-pytorch:example localhost:5000/triton-manylinux-pytorch:example
docker push localhost:5000/triton-manylinux-pytorch:example
```

然后给第 3 步的命令加上 PyTorch 相关参数（见下文），并在第 4 步补全服务镜像。`localhost:5000` 只是 NVIDIA 内部 registry 的替身。

## 第 3 步：构建服务器

克隆 server 仓库并切换到对应的发布分支，然后带上你的基础镜像运行 `build.py`：

```bash
git clone https://github.com/triton-inference-server/server.git
cd server && git checkout r26.05  # choose any release version

./build.py -v --target-platform=rhel --no-container-pull \
  --version=2.69.0 --container-version=26.05 --upstream-container-version=26.05 \
  --image=base,triton-manylinux-base:example \
  --image=pytorch,localhost:5000/triton-manylinux-pytorch:example \
  --extra-core-cmake-arg=PYBIND11_FINDPYTHON=ON \
  --enable-gpu --enable-logging --enable-stats --enable-metrics \
  --enable-gpu-metrics --enable-cpu-metrics --enable-tracing \
  --endpoint=http --endpoint=grpc \
  --backend=onnxruntime:r26.05 --backend=pytorch:r26.05 --backend=python:r26.05 \
  --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_NVSHMEM=ON \
  --extra-backend-cmake-arg=pytorch:TRITON_PYTORCH_ENABLE_TORCHVISION=OFF \
  --repoagent=checksum:r26.05
```

关键参数：

- `--no-container-pull` —— 假设你的基础镜像在本地，这个参数阻止 Docker 尝试拉取它。它**不会**阻止 pytorch 后端自己对 `--image=pytorch` 的拉取——这正是第 2 步要推送到本地 registry 的原因。
- 无头环境（CI、无 TTY 的 `ssh` 等）运行时加上 `--no-container-interactive`——build.py 默认用 `docker run -it` 启动编译，没有终端时会报 `the input device is not a TTY` 并中止。
- `--extra-core-cmake-arg=PYBIND11_FINDPYTHON=ON` —— 让 pybind11 使用你安装的 Python 3.12，而不是 Rocky 8 自带的系统 `python3`（3.6）；后者缺少开发头文件，会以 `fatal error: Python.h` 失败。
- `--image=pytorch,localhost:5000/…` —— 第 2 步构建好的 PyTorch 镜像。其他后端（onnxruntime、tensorrt、python）在构建期间从源码编译；PyTorch 则复用**预构建**的 libtorch（在构建树内编译太重），由 build.py `docker pull` 该镜像并从中解压——它是唯一需要 `--image` 的后端。
- `TRITON_PYTORCH_ENABLE_TORCHVISION=OFF` —— 本示例不构建 torchvision；在本教程中，从公开来源接线 torchvision 是一条未经验证的路径。
- `TRITON_PYTORCH_NVSHMEM=ON` —— 保持 nvshmem 开启，这样 build.py 会复制 `libtorch_nvshmem.so`（libtorch 链接了它）；第 4 步会补上它运行时所需的那个库。

这里构建 `python` 后端，是为了让 build.py 配好 pytorch 后端对外服务所需的 pyenv Python 和 numpy（第 4 步）。它是可选的——去掉 `--backend=python` 的话，需要自己在补全镜像中重新加回这些东西。`tensorrt` 也是可选的（ONNX Runtime 已自带 TensorRT provider）；需要独立后端时再加 `--backend=tensorrt:r26.05`。

Triton 的 `common`、`core`、`backend` 和 `third_party` 仓库不需要显式传 `--repo-tag` 参数——build.py 默认把它们固定到与 `--container-version`（此处为 `r26.05`）匹配的分支。

ONNX Runtime 在这里从源码编译（约 2 小时——要为多种 GPU 架构编译 CUDA kernel）。想加速的话，只为你 GPU 的架构构建——参见 [`onnxruntime_backend`](https://github.com/triton-inference-server/onnxruntime_backend) 的构建选项。

构建产物位于 `build/install/` 下，同时生成一个本地 `tritonserver` Docker 镜像。

## 第 4 步：验证

**1. manylinux 构件生成**

> [!NOTE]
> 当前 `rhel` 构建**产出的 wheel 标签是 `linux_x86_64`，而不是 `manylinux`。** 它*确实*会
> 运行 `auditwheel repair` 并生成正确的 `manylinux_2_27` wheel（位于构建容器的
> `.../python/generic/` 目录下），但打包步骤安装的是*未修复*的副本。变通办法：把修复好的 wheel 从构建容器中取出来：

```bash
docker start tritonserver_builder >/dev/null
docker exec tritonserver_builder sh -c 'find /tmp/tritonbuild -name "*manylinux*.whl"' \
  | while read -r w; do docker cp "tritonserver_builder:$w" build/install/python/; done
docker stop tritonserver_builder >/dev/null
find build/install/python -name '*manylinux*.whl' | grep -q . \
  || { echo "ERROR: no manylinux wheels extracted; leaving originals in place." >&2; exit 1; }
rm -f build/install/python/*-linux_x86_64.whl
```

现在 `ls` 一下构件，并用 `auditwheel` 证明标签货真价实（不只是文件名）——auditwheel 会检查 wheel 的外部符号是否真的落在目标 glibc 版本之内（auditwheel 会挑出真实的最低版本——这里是 `2_27`，比 2_28 *更*可移植）：

```bash
ls build/install/backends                 # onnxruntime  pytorch  python
find build/install -name '*.whl'          # ...-cp312-cp312-manylinux_2_27_x86_64.whl

# auditwheel lives in the base image — run it from there, nothing installed on your host:
docker run --rm -v "$PWD/build:/b:ro" triton-manylinux-base:example \
  bash -c 'auditwheel show /b/install/python/tritonserver-*.whl'
#  ... is consistent with the following platform tag: "manylinux_2_27_x86_64"
#  ... external versioned symbols in system libraries: libc.so.6 (GLIBC_2.2.5 ... 2.27)
```

**2. 从 manylinux 容器跑真实负载**

启动服务器并跑真实推理。服务器二进制链接的是 EL8 的 `libssl.so.1.1`，所以要在**构建出的镜像内**（Rocky 8）运行，而不是在非 EL8 宿主机上。先创建一个模型仓库，包含两个 `OUTPUT0 = INPUT0 + INPUT1` 模型——一个 Python 后端，一个 ONNX。

```bash
# python backend model
mkdir -p models/add_py/1
cat > models/add_py/config.pbtxt <<'EOF'
name: "add_py"
backend: "python"
max_batch_size: 0
input [
  { name: "INPUT0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT0", data_type: TYPE_FP32, dims: [4] } ]
instance_group [ { kind: KIND_CPU } ]
EOF
cat > models/add_py/1/model.py <<'EOF'
import numpy as np
import triton_python_backend_utils as pb_utils
class TritonPythonModel:
    def execute(self, requests):
        out = []
        for r in requests:
            a = pb_utils.get_input_tensor_by_name(r, "INPUT0").as_numpy()
            b = pb_utils.get_input_tensor_by_name(r, "INPUT1").as_numpy()
            t = pb_utils.Tensor("OUTPUT0", (a + b).astype(np.float32))
            out.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return out
EOF

# onnx model (validates the onnxruntime backend)
mkdir -p models/add_onnx/1
cat > models/add_onnx/config.pbtxt <<'EOF'
name: "add_onnx"
backend: "onnxruntime"
max_batch_size: 0
input [
  { name: "INPUT0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT0", data_type: TYPE_FP32, dims: [4] } ]
instance_group [ { kind: KIND_CPU } ]
EOF
# generate the .onnx in a temporary container (no host install)
cat > /tmp/gen_onnx.py <<'EOF'
import onnx
from onnx import helper, TensorProto
g = helper.make_graph(
    [helper.make_node("Add", ["INPUT0", "INPUT1"], ["OUTPUT0"])], "add",
    [helper.make_tensor_value_info("INPUT0", TensorProto.FLOAT, [4]),
     helper.make_tensor_value_info("INPUT1", TensorProto.FLOAT, [4])],
    [helper.make_tensor_value_info("OUTPUT0", TensorProto.FLOAT, [4])])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]),
          "/models/add_onnx/1/model.onnx")
EOF
docker run --rm -v "$PWD/models:/models" -v /tmp/gen_onnx.py:/gen.py:ro python:3.12-slim \
  bash -c "pip install --quiet onnx && python /gen.py"
```

启动服务器（仅 CPU，因此该检查不需要 GPU）。如果 `8000` 端口已被占用，就映射一个空闲的主机端口，例如 `-p 8080:8000`，然后使用下面的 `localhost:8080`：

```bash
docker run --rm -p8000:8000 -p8001:8001 -v "$PWD/models:/models" \
  tritonserver:latest tritonserver --model-repository=/models
# wait for: "successfully loaded 'add_py'" / "'add_onnx'" and "Started HTTPService"
```

在第二个终端中发送推理请求：

```bash
curl -s localhost:8000/v2/health/ready -o /dev/null -w "ready: %{http_code}\n"

for m in add_py add_onnx; do echo "== $m =="; curl -s localhost:8000/v2/models/$m/infer \
  -H 'Content-Type: application/json' -d '{
  "inputs":[
    {"name":"INPUT0","shape":[4],"datatype":"FP32","data":[1,2,3,4]},
    {"name":"INPUT1","shape":[4],"datatype":"FP32","data":[10,20,30,40]}]}'; echo; done
```

预期结果：`ready: 200`，两个模型都返回 `OUTPUT0 = [11, 22, 33, 44]`。`add_onnx` 结果正确，说明 `onnxruntime` 后端在重建的公共基础镜像上服务正常。用 `Ctrl‑C` 停止服务器。

> 💡 **AI Infra 视角**：manylinux 是 Python 生态"一次构建、跨发行版运行"的规范：wheel 的平台标签声明它能兼容的最低 glibc 版本（如 `manylinux_2_27`），auditwheel 通过检查二进制实际引用的 glibc 符号来验证标签是否属实。对企业部署而言，这意味着同一份 Triton wheel 可以同时部署到 RHEL 8、Rocky、AlmaLinux 等不同发行版，不用每换一个系统就重新编译一遍。

### PyTorch 后端

`pytorch` 后端要在**独立的模型仓库**（`models_torch/`，GPU）中**单独**验证。先补全服务镜像：[`Dockerfile.pytorch-runtime.rhel`](Dockerfile.pytorch-runtime.rhel) 会把 `torch` 安装到 `python` 后端已经配好的 pyenv Python 里：

```bash
docker build -f Dockerfile.pytorch-runtime.rhel -t tritonserver-pytorch:example .
```

加一个 TorchScript 模型 `OUTPUT__0 = INPUT__0 + INPUT__1`（PyTorch 后端使用 `INPUT__N` / `OUTPUT__N` 命名约定）：

```bash
mkdir -p models_torch/add_torch/1
cat > models_torch/add_torch/config.pbtxt <<'EOF'
name: "add_torch"
backend: "pytorch"
max_batch_size: 0
input [
  { name: "INPUT__0", data_type: TYPE_FP32, dims: [4] },
  { name: "INPUT__1", data_type: TYPE_FP32, dims: [4] }
]
output [ { name: "OUTPUT__0", data_type: TYPE_FP32, dims: [4] } ]
instance_group [ { kind: KIND_GPU } ]
EOF
# script the model in the Step 2 image (it already has torch)
cat > /tmp/gen_pt.py <<'EOF'
import torch
class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b
torch.jit.script(Add()).save("/models/add_torch/1/model.pt")
EOF
docker run --rm -v "$PWD/models_torch:/models" -v /tmp/gen_pt.py:/gen.py:ro \
  triton-manylinux-pytorch:example python /gen.py
```

用补全后的镜像对外服务（`KIND_GPU`，因此需要 GPU）并推理。（同样，如果 `8000` 被占用，映射一个空闲端口——`-p 8080:8000`，然后 `localhost:8080`。）

```bash
docker run --rm --gpus all -p8000:8000 -v "$PWD/models_torch:/models" \
  tritonserver-pytorch:example tritonserver --model-repository=/models
# wait for: "successfully loaded 'add_torch'", then in a second terminal:

curl -s localhost:8000/v2/models/add_torch/infer -H 'Content-Type: application/json' -d '{
  "inputs":[
    {"name":"INPUT__0","shape":[4],"datatype":"FP32","data":[1,2,3,4]},
    {"name":"INPUT__1","shape":[4],"datatype":"FP32","data":[10,20,30,40]}]}'; echo
```

预期结果：`OUTPUT__0 = [11, 22, 33, 44]`——完全从公开源码构建的 PyTorch 后端，推理结果正确。

## 与官方构件的已知差异

这次构建与官方 `manylinux` 发布版*等价*，但并不*完全相同*：

- **钉死版本以保证一致。** `BASE_IMAGE` 和 `TENSORRT_VERSION`（第 1 步）以及
  `--version` / `--container-version`（第 3 步）都必须与目标发布版一致。
  可以对照发布版的构件名（`…-cu132-cp312-manylinux_2_28-x86_64.zip`）和框架支持矩阵交叉核对。
- **cuDNN** 来自公共 CUDA 基础镜像，补丁版本可能比发布版用的稍新；如果需要完全一致，
  在 `Dockerfile.base.rhel` 中从 `cuda-rhel8` 仓库安装指定版本的 cuDNN RPM。
- **运行二进制**需要 EL8 的 `libssl.so.1.1`。请在构建出的镜像内 / EL8 主机上运行，
  或者把 OpenSSL 1.1 随可执行文件一起打包。

关于 `build.py` 及其参数的背景知识，参见 server 仓库的
[构建文档](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md)。
