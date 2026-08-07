# ============================================================================
# Triton Python backend 的 tokenizer（分词）模型：
# 接收文本输入，用 HuggingFace BERT tokenizer 分词后输出
# input_ids / token_type_ids / attention_mask 张量，供下游模型消费。
# ============================================================================

from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import BertTokenizerFast, TensorType


# Triton Python backend 模型入口类
class TritonPythonModel:
    tokenizer: BertTokenizerFast

    # 模型加载时调用一次：加载 bert-base-cased 分词器（首次运行会下载权重）
    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    # 推理入口：对每个请求的 TEXT 输入做分词，返回分词结果张量
    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # 逐个处理请求（示例中未启用动态批处理）
        for request in requests:
            # 把二进制数据解码回字符串
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]
            # 分词：统一 padding 到 256 长度（超出截断），返回 numpy 张量
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query,
                return_tensors=TensorType.NUMPY,
                padding="max_length",
                max_length=256,
                truncation=True,
            )
            # TensorRT 用 int32 作输入类型，ONNX Runtime 用 int64，这里统一转 int64
            tokens = {k: v.astype(np.int64) for k, v in tokens.items()}
            # 把分词结果包装成 Triton 张量，回传给服务器
            outputs = list()
            # 按 model_input_names（input_ids、token_type_ids、attention_mask）逐个构造输出
            for input_name in self.tokenizer.model_input_names:
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    # 模型卸载时调用一次（可选实现）：用于清理资源
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
