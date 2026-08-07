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

import argparse
import sys
import time

import tritonserver

# 脚本用途：用 Triton 的 Python 客户端 API 加载模型仓库中的模型，
# 触发模型的 initialize 流程（对 diffusion 模型而言即构建 TensorRT 引擎），
# 加载成功后卸载模型，从而实现"构建引擎但不常驻服务"的批量建引擎流程。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 指定要构建的模型名，默认 "all" 表示构建仓库中的全部模型
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument(
        "--model-repository", type=str, default="/workspace/diffusion-models"
    )
    # 等待模型加载（引擎构建）的超时时间，默认 20 分钟
    parser.add_argument("--timeout", type=int, default=60 * 20)

    args = parser.parse_args()

    # 创建 Triton 服务器对象；EXPLICIT 模式下模型需显式调用 load 才会加载
    server = tritonserver.Server(
        model_repository=args.model_repository,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    )

    # 启动服务器并等待其进入就绪状态
    server.start(wait_until_ready=True)
    # 查询模型仓库中已注册的模型列表（键为 (模型名, 版本号)）
    models = server.models()

    if args.model == "all":
        models = models.keys()
    else:
        # 指定单个模型时，构造 (模型名, -1) 以匹配仓库中的元数据；-1 代表最新版本
        args.model = (args.model, -1)
        if not args.model in models:
            print(f"Model: {args.model} not known")
            sys.exit(1)
        models = [args.model]

    for model in models:
        # 跳过带具体版本号的条目，只处理最新版本
        if model[1] != -1:
            continue
        print(f"Loading Model: {model}")
        # 显式加载模型，触发引擎构建（或加载已有引擎）
        model = server.load(model[0])
        start = time.time()
        # 轮询等待模型就绪，直到超时
        while not model.ready() and ((time.time() - start) <= args.timeout):
            time.sleep(10)

        if model.ready():
            print(f"Model: {model} Loaded")
        else:
            print(f"Error loading: {model}")
            sys.exit(1)

        # 构建完成后卸载模型，释放显存
        server.unload(model, wait_until_unloaded=True)

    server.stop()
