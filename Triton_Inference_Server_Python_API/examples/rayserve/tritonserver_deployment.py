# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ============================================================================
# Ray Serve 示例的核心部署文件，提供两个可切换的 deployment：
#   * TritonDeployment：进程内 Triton server 加载 identity 与 stable diffusion
#     模型，FastAPI 暴露 /identity（字符串原样返回）与 /generate（文生图）端点；
#   * BaseDeployment：不经过 Triton，直接用 diffusers 管线生成图片的基线实现，
#     用于对比「直接推理」与「经 Triton 推理」的性能差异。
# ============================================================================

import os
from pprint import pprint
from typing import Optional

import numpy
import requests
import torch
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve

# 1: 定义一个 FastAPI 应用，供后面的 deployment 通过 @serve.ingress 挂载路由
app = FastAPI()

# 可选：从环境变量读取 S3 桶地址，把模型仓库放到远端
S3_BUCKET_URL = None

if "S3_BUCKET_URL" in os.environ:
    S3_BUCKET_URL = os.environ["S3_BUCKET_URL"]


# 打印带分隔线的小标题，便于在日志中定位关键节点
def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


# 基线部署：直接用 diffusers 的 Stable Diffusion XL 管线生成图片（不经 Triton），
# 用于与 Triton 版本对比性能；申请 1 块 GPU，按在途请求数自动扩缩容（1~8 个副本）
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "max_ongoing_requests": 1,
        "target_ongoing_requests": 1,
        "upscale_delay_s": 2,
        "downscale_delay_s": 120,
        "upscaling_factor": 1,
        "downscaling_factor": 1,
        "metrics_interval_s": 2,
        "look_back_period_s": 4,
    },
)
@serve.ingress(app)
class BaseDeployment:
    def __init__(self, use_torch_compile=False):
        self._image_size = 1024
        self._model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        from diffusers import AutoencoderKL, DiffusionPipeline

        # 加载 SDXL 管线：fp16 精度 + 修复版 VAE（解决显存与画质问题）
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        self._pipeline = DiffusionPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            vae=vae,
        )
        # 把管线整体搬到 GPU
        self._pipeline = self._pipeline.to("cuda")
        # 可选：用 torch.compile 编译 UNet 部分以加速
        if use_torch_compile:
            print("compiling")
            print(torch._dynamo.list_backends())
            self._pipeline.unet = torch.compile(
                self._pipeline.unet,
                fullgraph=True,
                mode="reduce-overhead",
                dynamic=False,
            )
        # 预热：先跑一次生成，触发 CUDA kernel 与显存分配
        self.generate("temp")

    # HTTP GET /generate：文生图，可选保存到文件
    @app.get("/generate")
    def generate(self, prompt: str, filename: Optional[str] = None) -> None:
        with torch.autocast("cuda"):
            image_ = self._pipeline(
                prompt,
                height=self._image_size,
                width=self._image_size,
                num_inference_steps=30,
            ).images[0]
            if filename:
                image_.save(filename)


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "max_ongoing_requests": 1,
        "target_ongoing_requests": 1,
        "upscale_delay_s": 2,
        "downscale_delay_s": 120,
        "upscaling_factor": 1,
        "downscaling_factor": 1,
        "metrics_interval_s": 2,
        "look_back_period_s": 4,
    },
)
@serve.ingress(app)
# Triton 部署：进程内 Triton server 加载 identity 与 stable diffusion 模型，
# 通过 FastAPI 暴露 /identity 与 /generate 端点；同样启用自动扩缩容
class TritonDeployment:
    def __init__(self):
        self._triton_server = tritonserver

        # 模型仓库：优先使用 S3 地址，否则使用本地两个模型目录
        if S3_BUCKET_URL is not None:
            model_repository = S3_BUCKET_URL
        else:
            model_repository = [
                "/workspace/identity-models",
                "/workspace/diffusion-models",
            ]

        # 创建进程内 Triton server：显式模型控制模式（模型需手动 load）
        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        # 阻塞等待 server 完全就绪
        self._triton_server.start(wait_until_ready=True)

        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        self._stable_diffusion = None
        self._test_model = None

        # 显式加载 stable diffusion 模型，并确认其进入 READY 状态
        if not self._triton_server.model("stable_diffusion_xl").ready():
            try:
                self._stable_diffusion = self._triton_server.load("stable_diffusion_xl")

                if not self._stable_diffusion.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't load stable diffusion model!")
                print(
                    f"Please ensure dependencies are met and you have set the environment variable HF_TOKEN {error}"
                )
                return
        _print_heading("Models")
        pprint(self._triton_server.models())
        # 预热：先跑一次生成
        self.generate("temp")

    # HTTP GET /identity：把字符串输入原样返回（用于验证整条链路可用）
    @app.get("/identity")
    def test(self, string_input: str) -> str:
        # identity 模型懒加载：首次调用时才 load
        if not self._triton_server.model("identity").ready():
            self._test_model = self._triton_server.load("identity")

        output = []
        # 调用 Triton 推理：输入 shape 为 [[string_input]]
        for response in self._test_model.infer(
            inputs={"string_input": [[string_input]]}
        ):
            # 从响应中取 string_output，逐条拼成最终字符串
            output.append(response.outputs["string_output"].to_string_array()[0][0])

        return "".join(output)

    # HTTP GET /generate：文生图，可选保存到文件
    @app.get("/generate")
    def generate(self, prompt: str, filename: Optional[str] = None) -> None:
        # 调用 Triton 中 stable diffusion 模型推理，返回生成的图像张量
        for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
            # 通过 DLPack 把 GPU 张量转成 numpy（零拷贝），再转为 PIL 图片
            generated_image = (
                numpy.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(numpy.uint8)
            )

            image_ = Image.fromarray(generated_image)
            if filename:
                image_.save(filename)


# serve run 的默认入口：部署 Triton 版本（可改用 baseline 对比性能）
def deployment(_args):
    return TritonDeployment.bind()


# 基线版本入口：可选 use-torch-compile 参数
def baseline(_args):
    if "use-torch-compile" in _args:
        return BaseDeployment.bind(use_torch_compile=True)
    else:
        return BaseDeployment.bind(use_torch_compile=False)


if __name__ == "__main__":
    # 2: 部署 TritonDeployment，并挂载到根路径
    serve.run(TritonDeployment.bind(), route_prefix="/")

    # 3: 向部署发起请求并打印结果
    print(
        requests.get(
            "http://localhost:8000/identity", params={"name": "Theodore"}
        ).json()
    )

    # 3: 向部署发起请求并打印结果
    print(
        requests.get(
            "http://localhost:8000/generate",
            params={"prompt": "pigeon in new york, realistic, 4k, photograph"},
        )
    )
