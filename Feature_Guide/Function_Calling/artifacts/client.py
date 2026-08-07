#!/usr/bin/python
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
import json
import sys

import client_utils
import numpy as np
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="Inference server URL."
    )

    parser.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt.")

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="ensemble",
        choices=["ensemble", "tensorrt_llm_bls"],
        help="Name of the Triton model to send request to",
    )

    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=None,
        help="The repetition penalty value",
    )

    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )

    parser.add_argument(
        "--frequency-penalty",
        type=float,
        required=False,
        default=None,
        help="The frequency penalty value",
    )

    parser.add_argument(
        "-o",
        "--output-len",
        type=int,
        default=100,
        required=False,
        help="Specify output length",
    )

    parser.add_argument(
        "--request-id",
        type=str,
        default="",
        required=False,
        help="The request_id for the stop request",
    )

    parser.add_argument("--stop-words", nargs="+", default=[], help="The stop words")

    parser.add_argument("--bad-words", nargs="+", default=[], help="The bad words")

    parser.add_argument(
        "--embedding-bias-words", nargs="+", default=[], help="The biased words"
    )

    parser.add_argument(
        "--embedding-bias-weights",
        nargs="+",
        default=[],
        help="The biased words weights",
    )

    parser.add_argument(
        "--overwrite-output-text",
        action="store_true",
        required=False,
        default=False,
        help="In streaming mode, overwrite previously received output text instead of appending to it",
    )

    parser.add_argument(
        "--return-context-logits",
        action="store_true",
        required=False,
        default=False,
        help="Return context logits, the engine must be built with gather_context_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--return-generation-logits",
        action="store_true",
        required=False,
        default=False,
        help="Return generation logits, the engine must be built with gather_ generation_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--end-id", type=int, required=False, help="The token id for end token."
    )

    parser.add_argument(
        "--pad-id", type=int, required=False, help="The token id for pad token."
    )

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8001"

    embedding_bias_words = (
        FLAGS.embedding_bias_words if FLAGS.embedding_bias_words else None
    )
    embedding_bias_weights = (
        FLAGS.embedding_bias_weights if FLAGS.embedding_bias_weights else None
    )

    # 创建 gRPC 客户端（默认端口 8001），连接失败则退出
    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    return_context_logits_data = None
    if FLAGS.return_context_logits:
        return_context_logits_data = np.array(
            [[FLAGS.return_context_logits]], dtype=bool
        )

    return_generation_logits_data = None
    if FLAGS.return_generation_logits:
        return_generation_logits_data = np.array(
            [[FLAGS.return_generation_logits]], dtype=bool
        )

    # 组装系统提示词：注入工具定义与函数调用 JSON schema
    prompt = client_utils.process_prompt(FLAGS.prompt)

    # 实例化工具实现（get_current_stock_price / get_company_news / final_answer）
    functions = client_utils.MyFunctions()

    # 函数调用主循环：LLM 每轮返回一次工具调用请求，客户端执行后把结果回填 prompt 再继续，
    # 直到 LLM 调用 final_answer（或达到第 50 步）才结束
    while True:
        output_text = client_utils.run_inference(
            client,
            prompt,
            FLAGS.output_len,
            FLAGS.request_id,
            FLAGS.repetition_penalty,
            FLAGS.presence_penalty,
            FLAGS.frequency_penalty,
            FLAGS.temperature,
            FLAGS.stop_words,
            FLAGS.bad_words,
            embedding_bias_words,
            embedding_bias_weights,
            FLAGS.model_name,
            FLAGS.streaming,
            FLAGS.beam_width,
            FLAGS.overwrite_output_text,
            return_context_logits_data,
            return_generation_logits_data,
            FLAGS.end_id,
            FLAGS.pad_id,
            FLAGS.verbose,
        )

        # 解析 LLM 返回的 JSON；格式非法则报错退出（可用约束解码从源头避免）
        try:
            response = json.loads(output_text)
        except ValueError:
            print("\n[ERROR] LLM responded with invalid JSON format!")
            break

        # Repeat the loop until `final_answer` tool is called, which indicates
        # that the full response is ready and llm does not require any
        # additional information. Additionally, if the loop has taken more
        # than 50 steps, the script ends.
        # 终止条件：调用了 final_answer（输出最终答案）或超过 50 步上限
        if response["tool"] == "final_answer" or response["step"] == "50":
            if response["tool"] == "final_answer":
                final_response = response["arguments"]["final_response"]
                print("\n\n+++++++++++++++++++++++++++++++++++++")
                print(f"RESPONSE: {final_response}")
                print("+++++++++++++++++++++++++++++++++++++\n\n")
            elif response["step"] == "50":
                print("\n\n+++++++++++++++++++++++++++++++++++++")
                print(f"Reached maximum number of function calls available.")
                print("+++++++++++++++++++++++++++++++++++++\n\n")
            break

        # Extract tool's name and arguments from the response
        # 从响应中取出工具名和参数，动态调用对应的函数实现
        function_name = response["tool"]
        function_args = response["arguments"]
        function_to_call = getattr(functions, function_name)
        # Execute function call and store results in `function_response`
        function_response = function_to_call(*function_args.values())

        if FLAGS.verbose:
            print("=====================================")
            print(f"Executing function: {function_name}({function_args}) ")
            print(f"Function response: {str(function_response)}")
            print("=====================================")

        # Update prompt with the generated function call and results of that
        # call.
        # 把「工具调用 + 工具返回结果」以 tool_response 形式回填 prompt，进入下一轮对话
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        prompt += str(
            output_text
            + "<|im_end|>\n<tool_response>"
            + str(results_dict)
            + "</tool_response>\n<|im_start|>assistant"
        )
