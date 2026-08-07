# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


# 主流程：发送文本提示词，调用 pipeline 模型生成图片并保存
def main():
    # 建立与 Triton 的 HTTP 连接
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # 文本提示词需要按字符串数组（object dtype）组织成 [1, 1] 形状
    prompt = "Pikachu with a hat, 4k, 3d render"
    text_obj = np.array([prompt], dtype="object").reshape((-1, 1))

    # 构造输入对象：输入名 "prompt" 与 pipeline 模型的配置一致
    input_text = httpclient.InferInput(
        "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
    )
    input_text.set_data_from_numpy(text_obj)

    # 声明要获取的输出张量
    output_img = httpclient.InferRequestedOutput("generated_image")

    # 一次请求即可驱动服务端完整跑完 文本编码 → 迭代去噪 → 解码 的流水线
    query_response = client.infer(
        model_name="pipeline", inputs=[input_text], outputs=[output_img]
    )

    # 取出生成的图片张量并保存为 jpg
    image = query_response.as_numpy("generated_image")
    im = Image.fromarray(np.squeeze(image.astype(np.uint8)))
    im.save("generated_image2.jpg")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print("Time taken:", end - start)
