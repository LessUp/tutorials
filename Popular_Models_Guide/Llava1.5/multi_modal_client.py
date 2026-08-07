# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import queue
import warnings
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

warnings.filterwarnings("ignore")


# 异步流式推理的辅助类：通过队列收集回调返回的结果或错误
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# 异步推理回调：把每个完成的结果（或错误）放入 UserData 队列
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


# 构造 Triton gRPC 输入张量：指定名称、形状，并把 numpy 数据装载进去
def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="llava-1.5",
        help="Model name",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        required=False,
        default="http://images.cocodataset.org/test2017/000000557146.jpg",
        help="Image URL. Default is:\
                            http://images.cocodataset.org/test2017/000000557146.jpg",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default="What is shown on the picture?",
        help="Prompt. Default is:\
                            What is shown on the picture?",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=False,
        default=50,
        help="Max amount of tokens in the output. Default is 50.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.9,
        help="Temperatue. Default is 0.9.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=1,
        help="Top K. Default is 1.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        required=False,
        default=0.9,
        help="Frequency penalty. Default is 0.9.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=10,
        help="Random seed. Default is 10.",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )

    args = parser.parse_args()
    user_data = UserData()

    # 按 Llava 的对话模板拼接提示词：<image> 占位符表示图片特征将插入的位置
    input_text = "USER: <image>\nQuestion:" + args.prompt + " Answer:"
    # 图片以 URL 字符串形式传入，由服务端拉取；各参数按类型封装为 numpy 数组
    image_url = np.array([args.image_url.encode("utf-8")], dtype=np.object_)
    prompt_data = np.array([input_text.encode("utf-8")], dtype=np.object_)
    max_tokens = np.array([args.max_tokens], dtype=np.int32)
    temperature = np.array([args.temperature], dtype=np.float32)
    top_k = np.array([args.top_k], dtype=np.int32)
    frequency_penalty = np.array([args.frequency_penalty], dtype=np.float32)
    seed = np.array([args.seed], dtype=np.uint64)
    # 组装全部输入张量：image、prompt 及各项采样参数
    inputs = [
        prepare_tensor("image", image_url),
        prepare_tensor("prompt", prompt_data),
        prepare_tensor("max_tokens", max_tokens),
        prepare_tensor("temperature", temperature),
        prepare_tensor("top_k", top_k),
        prepare_tensor("frequency_penalty", frequency_penalty),
        prepare_tensor("seed", seed),
    ]
    # 声明期望的输出：生成文本及 token 统计信息
    outputs = []
    for output_name in [
        "text",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]:
        outputs.append(grpcclient.InferRequestedOutput(output_name))
    output_text = ""

    # 通过 gRPC（8001 端口）建立流式推理客户端：
    # 流式模式下结果通过回调异步到达，适合长时间生成或流式输出的场景
    with grpcclient.InferenceServerClient(url="localhost:8001") as client:
        client.start_stream(partial(callback, user_data))
        client.async_stream_infer(
            args.model_name,
            inputs,
            outputs=outputs,
        )
        expected_responses = 1
        processed_count = 0
        # 从队列中取出流式响应并处理，直到收到预期数量的响应
        while processed_count < expected_responses:
            try:
                result = user_data._completed_requests.get()
                print("Got completed request", flush=True)
            except Exception:
                break

            # 出错时区分"请求被取消"与其他服务端错误
            if type(result) == InferenceServerException:
                if result.status() == "StatusCode.CANCELLED":
                    print("Request is cancelled")
                else:
                    print("Received an error from server:")
                    print(result)
                    raise result
            else:
                # 正常响应：读取生成的文本并打印
                output_text = result.as_numpy("text")
                print(output_text[0].decode("utf-8"))

            processed_count = processed_count + 1

            client.stop_stream()
