# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Ray Serve 示例的压测客户端：通过多进程并发向部署端点发送请求，
# 统计吞吐（throughput）与平均延迟，也可用 nvidia-smi 记录 GPU 使用情况。
# ============================================================================

import argparse
import os
import subprocess
import time
from multiprocessing import Process

import numpy as np
import requests
from tqdm import tqdm

file_location = os.path.dirname(os.path.realpath(__file__))


# 单个客户端的压测逻辑：连续发送 request_count 次请求，统计每次延迟
def client(endpoint, request_count, prompt, save_image, index):
    latencies = []
    start = time.time()
    for i in tqdm(range(request_count)):
        # 可选：把生成的图片保存到本地文件（按客户端序号与请求序号命名）
        if save_image:
            filename = os.path.join(
                file_location, f"client_{index}_generated_image{i}.jpg"
            )
            # URL 查询参数中的空格需编码为 %20
            filename_input = "%20".join(f"&filename={filename}".split(" "))
        else:
            filename_input = ""
        prompt_input = "%20".join(prompt.split(" "))
        request_start = time.time()
        # 向 Ray Serve 部署端点发起同步 HTTP 请求
        requests.get(
            f"http://127.0.0.1:8000/{endpoint}?prompt={prompt_input}{filename_input}",
            timeout=300,
        )
        latencies.append(time.time() - request_start)
    # 输出该客户端的吞吐与平均延迟
    print(
        f"Client: {index} Throughput: {request_count/(time.time()-start)} Avg. Latency: {np.mean(latencies)}"
    )


if __name__ == "__main__":
    # 命令行参数：客户端数、每客户端请求数、提示词、是否保存图片、是否记录 GPU 状态、端点
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument(
        "--prompt",
        type=str,
        default="skeleton sitting by the side of a river looking soulful, concert poster, 4k, artistic",
    )
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--launch-nvidia-smi", action="store_true")
    parser.add_argument("--endpoint", type=str, default="generate")
    args = parser.parse_args()
    # 可选：启动 nvidia-smi dmon 在后台持续记录 GPU 指标到文件
    if args.launch_nvidia_smi:
        nvidia_smi_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-f", "nvidia_smi_output.txt"]
        )
        time.sleep(5)
    procs = []
    start_time = time.time()
    # 每个客户端启动一个独立进程，并发发起压测
    for i in range(args.clients):
        procs.append(
            Process(
                target=client,
                args=(
                    args.endpoint,
                    args.requests,
                    args.prompt,
                    args.save_image,
                    i,
                ),
            )
        )
        procs[-1].start()

    # 等待所有客户端进程结束，汇总计算总吞吐
    for proc in procs:
        proc.join()
    end_time = time.time()
    if args.launch_nvidia_smi:
        time.sleep(5)
        nvidia_smi_proc.kill()
    print(
        f"Throughput: {(args.requests*args.clients)/(end_time-start_time)} Total Time: {end_time-start_time}"
    )
