<!--
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton Inference Server 中的约束解码（Constrained Decoding）

本教程聚焦约束解码（constrained decoding）——一项确保大语言模型（LLM）输出严格遵守格式要求的重要技术。这些格式要求仅靠微调来满足，可能既困难又昂贵。

> 💡 **AI Infra 视角**：约束解码在推理服务中的核心价值是「把结构化输出的可靠性从「模型发挥」变成「系统保证」。在 Agent、RAG 等生产场景里，下游解析器（如 JSON parser）对输出格式是硬依赖，一次格式错误就可能导致整个调用链失败或触发重试。在服务端做约束解码，可以在采样阶段直接从源头排除非法 token，比「生成后校验重试」省一次完整生成，同时显著降低 TTFT/TPOT 的无效波动。

## 目录

- [约束解码简介](#约束解码简介)
- [前置条件：Hermes-2-Pro-Llama-3-8B](#前置条件hermes-2-pro-llama-3-8b)
- [通过提示工程实现结构化生成](#通过提示工程实现结构化生成)
    * [示例 1](#示例-1)
    * [示例 2](#示例-2)
- [通过外部库强制输出格式](#通过外部库强制输出格式)
    * [前置条件：通用配置](#前置条件通用配置)
        + [Logits 后处理器](#logits-后处理器)
        + [分词器（Tokenizer）](#分词器tokenizer)
        + [仓库配置](#仓库配置)
    * [LM Format Enforcer](#lm-format-enforcer)
    * [Outlines](#outlines)

## 约束解码简介

约束解码是自然语言处理和各种 AI 应用中用于引导和控制模型输出的一项强大技术。通过施加特定约束，该方法能确保生成输出符合预设标准，例如长度、格式或内容限制。在规则必须遵守的场景中（如生成合法代码片段、结构化数据或语法正确的句子），这一能力不可或缺。

在最近的发展中，一些模型已经通过微调将这些约束内化到模型中。这些模型在生成过程中能无缝整合约束，从而减少对大量后处理的需求，提升需要严格遵循预设规则的任务的效率和准确性。这种内置能力使它们在自动化内容创作、数据校验和实时语言翻译等对精度和可靠性要求极高的应用中特别有价值。

本教程基于 [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 模型，它已经支持 JSON 结构化输出（JSON Structured Outputs）。关于如何用 Triton Inference Server 和 TensorRT-LLM backend 部署 Hermes-2-Pro-Llama-3-8B 的详细教程，参见[这个](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)教程。这类场景下，输出的结构和质量可以通过提示工程（prompt engineering）来控制。想了解这条路径，请参考本教程的[通过提示工程实现结构化生成](#通过提示工程实现结构化生成)一节。

对于没有针对约束解码做过微调的模型，或者需要对输出进行更精确控制的场景，*LM Format Enforcer*（[GitHub](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)）和 *Outlines*（[GitHub](https://github.com/outlines-dev/outlines?tab=readme-ov-file)）等专用库提供了可靠的解决方案。这些库提供了对模型输出施加特定约束的工具，让开发者能够定制生成过程以满足精确需求。借助这些库，用户可以对输出获得更强的控制力，确保输出与期望标准完全一致——无论是保持特定格式、遵守内容规范，还是保证语法正确。本教程将演示如何在自己的工作流中使用 *LM Format Enforcer* 和 *Outlines*。

## 前置条件：Hermes-2-Pro-Llama-3-8B

在继续之前，请确保你已经按照[这些步骤](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)用 Triton Inference Server 和 TensorRT-LLM backend 成功部署了 [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 模型。

## 通过提示工程实现结构化生成

首先，启动 Triton SDK 容器：
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials:/tutorials \
    -v /path/to/Hermes-2-Pro-Llama-3-8B/repo:/Hermes-2-Pro-Llama-3-8B \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

提供的客户端脚本用到了 `pydantic` 库，SDK 容器中没有自带。继续之前请先安装：

```bash
pip install pydantic
```

### 示例 1

对于微调过的模型，只需构造如下的系统提示（system prompt）即可开启 JSON 模式：

```
You are a helpful assistant that answers in JSON.
```
完整的 `prompt` 组装逻辑请参考 [`client.py`](./artifacts/client.py)。

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200 --use-system-prompt
```
你应该会收到类似下面的响应：

```
...
assistant
{
  "title": "Harry Potter and the Order of Phoenix",
  "book_number": 5,
  "author": "J.K. Rowling",
  "series": "Harry Potter",
  "publication_date": "June 21, 2003",
  "page_count": 766,
  "publisher": "Arthur A. Levine Books",
  "genre": [
    "Fantasy",
    "Adventure",
    "Young Adult"
  ],
  "awards": [
    {
      "award_name": "British Book Award",
      "category": "Children's Book of the Year",
      "year": 2004
    }
  ],
  "plot_summary": "Harry Potter and the Order of Phoenix is the fifth book in the Harry Potter series. In this installment, Harry returns to Hogwarts School of Witchcraft and Wizardry for his fifth year. The Ministry of Magic is in denial about the return of Lord Voldemort, and Harry finds himself battling against the

```

### 示例 2

（可选）我们还可以把输出限制为特定的 schema。例如，在 [`client.py`](./artifacts/client.py) 中我们用 `pydantic` 库定义了如下回答格式：

```python
from pydantic import BaseModel

class AnswerFormat(BaseModel):
    title: str
    year: int
    director: str
    producer: str
    plot: str

...

prompt += "Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>".format(
                schema=AnswerFormat.model_json_schema())

```
来试试：

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Give me information about Harry Potter and the Order of Phoenix" -o 200 --use-system-prompt --use-schema
```
你应该会收到类似下面的响应：

```
 ...
assistant
{
  "title": "Harry Potter and the Order of Phoenix",
  "year": 2007,
  "director": "David Yates",
  "producer": "David Heyman",
  "plot": "Harry Potter and his friends must protect Hogwarts from a threat when the Ministry of Magic is taken over by Lord Voldemort's followers."
}

```

## 通过外部库强制输出格式

本小节将演示如何对没有针对约束解码做过微调的 LLM 施加约束。*LM Format Enforcer*（[GitHub](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file)）和 *Outlines*（[GitHub](https://github.com/outlines-dev/outlines?tab=readme-ov-file)）提供了可靠的解决方案。

两个库的参考实现都放在 [`utils.py`](./artifacts/utils.py) 脚本中，该脚本还定义了输出格式 `AnswerFormat`：

```python
class WandFormat(BaseModel):
        wood: str
        core: str
        length: float

class AnswerFormat(BaseModel):
        name: str
        house: str
        blood_status: str
        occupation: str
        alive: str
        wand: WandFormat
```

> 💡 **AI Infra 视角**：提示工程与外部库的取舍本质上是在「零改造成本」和「强格式保证」之间权衡。提示工程只是「引导」——模型仍有可能输出非法格式，需要客户端再做校验兜底；而 LM Format Enforcer / Outlines 这类库则把 JSON schema 编译成有限状态机（FSM）或正则表达式，在每个采样步用掩码（mask）把不合法 token 的 logits 置为负无穷，从机制上杜绝非法输出。代价是每步多一次掩码计算和 tokenizer 前缀匹配的开销（通常可忽略），但对低延迟要求极高的场景仍需压测确认。

### 前置条件：通用配置

确保你已经按照[这些步骤](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)用 Triton Inference Server 和 TensorRT-LLM backend 成功部署了 Hermes-2-Pro-Llama-3-8B 模型。
> [!IMPORTANT]
> 启动 docker 容器时，确保 `tutorials` 文件夹挂载到了 `/tutorials`。

配置成功后，你应当有 `/opt/tritonserver/inflight_batcher_llm` 文件夹，并能成功发起几个推理请求（例如[示例 1](#示例-1)或[示例 2](#示例-2)中的请求）。

我们接下来会对模型文件做一些调整，如果你的服务器正在运行，可以通过以下命令停掉它：
```bash
pkill tritonserver
```

#### Logits 后处理器

这两个库都会在每一步生成时限制可选 token 的集合。在 TensorRT-LLM 中，用户可以自定义一个 [logits 后处理器（logits post-processor）](https://nvidia.github.io/TensorRT-LLM/latest/features/sampling.html#logits-processor)来掩蔽当前生成步骤中不应使用的 logits。

> 💡 **AI Infra 视角**：logits 后处理器是 TensorRT-LLM 把「采样策略」与「业务约束」解耦的关键挂载点：服务端在 Executor 配置里注册自定义处理器，客户端只需在请求里带上处理器名字（`logits_post_processor_name`），同一个模型就能按请求动态切换不同的约束策略（比如一个是 JSON schema，一个是正则），无需重新部署模型。这在多租户或多业务共用一个模型的场景里很实用。

对于通过 `python` backend 部署的 TensorRT-LLM 模型（即在 `tensorrt_llm/config.pbtxt` 中把 [`triton_backend`](https://github.com/NVIDIA/TensorRT-LLM/blob/97ab014bdbd2b20c567f1b63fb86c18b55aac661/triton_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt#L28C10-L28C29) 设为 `python` 时，Triton 的 python backend 会用 [`model.py`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/triton_backend/all_models/inflight_batcher_llm/tensorrt_llm/1/model.py) 来托管你的 TensorRT-LLM 模型），自定义 logits 处理器需要在模型初始化时作为 [Executor](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html#executor-api) 配置的一部分（[`logits_post_processor_map`](https://github.com/NVIDIA/TensorRT-LLM/blob/32ed92e4491baf2d54682a21d247e1948cca996e/tensorrt_llm/hlapi/llm_utils.py#L205)）来指定。下面是参考示例。

```diff
...

+ executor_config.logits_post_processor_map = {
+            "<custom_logits_processor_name>": custom_logits_processor
+           }
self.executor = trtllm.Executor(model_path=...,
                                model_type=...,
                                executor_config=executor_config)
...
```

另外，如果你想为每个请求单独启用 logits 后处理器，可以通过一个额外的 `input` 参数实现。例如，本教程在 `inflight_batcher_llm/tensorrt_llm/config.pbtxt` 中添加 `logits_post_processor_name`：
```diff
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
    allow_ragged_batch: true
  },
  ...
  {
    name: "lora_config"
	data_type: TYPE_INT32
	dims: [ -1, 3 ]
	optional: true
	allow_ragged_batch: true
- }
+ },
+ {
+   name: "logits_post_processor_name"
+   data_type: TYPE_STRING
+   dims: [ -1 ]
+   optional: true
+ }
]
...
```
并在 `inflight_batcher_llm/tensorrt_llm/1/model.py` 的 `execute` 函数中处理它：

```diff
def execute(self, requests):
    """`execute` must be implemented in every Python model. `execute`
    function receives a list of pb_utils.InferenceRequest as the only
    argument. This function is called when an inference is requested
    for this model.
    Parameters
    ----------
    requests : list
      A list of pb_utils.InferenceRequest
    Returns
    -------
    list
      A list of pb_utils.InferenceResponse. The length of this list must
      be the same as `requests`
    """
    ...

    for request in requests:
        response_sender = request.get_response_sender()
        if get_input_scalar_by_name(request, 'stop'):
            self.handle_stop_request(request.request_id(), response_sender)
        else:
            try:
                converted = convert_request(request,
                                            self.exclude_input_from_output,
                                            self.decoupled)
+               logits_post_processor_name = get_input_tensor_by_name(request, 'logits_post_processor_name')
+               if logits_post_processor_name is not None:
+                   converted.logits_post_processor_name = logits_post_processor_name.item().decode('utf-8')
            except Exception as e:
            ...
```
本教程中，Hermes-2-Pro-Llama-3-8B 模型是以 ensemble 的形式部署的。这意味着请求先由 `ensemble` 模型处理，然后依次发送给 `pre-processing` 模型、`tensorrt-llm` 模型，最后是 `post-processing` 模型。这个流程以及输入输出映射定义在 `inflight_batcher_llm/ensemble/config.pbtxt` 中。因此我们还需要同步更新 `inflight_batcher_llm/ensemble/config.pbtxt`，让 `ensemble` 模型把额外的输入参数正确传递给 `tensorrt-llm` 模型：

```diff
input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  },
  ...
  {
      name: "embedding_bias_weights"
      data_type: TYPE_FP32
      dims: [ -1 ]
      optional: true
- }
+ },
+ {
+   name: "logits_post_processor_name"
+   data_type: TYPE_STRING
+   dims: [ -1 ]
+   optional: true
+ }
]
output [
    ...
]
ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
    ...
    },
    {
      model_name: "tensorrt_llm"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "_INPUT_ID"
      }
      ...
      input_map {
        key: "bad_words_list"
        value: "_BAD_WORDS_IDS"
      }
+     input_map {
+       key: "logits_post_processor_name"
+       value: "logits_post_processor_name"
+     }
      output_map {
        key: "output_ids"
        value: "_TOKENS_BATCH"
      }
      ...
    }
    ...
```

如果你跟随本教程操作，请确保 `/opt/tritonserver/inflight_batcher_llm` 仓库中对应的文件做了相同的修改。

#### 分词器（Tokenizer）

[*LM Format Enforcer*](https://github.com/noamgat/lm-format-enforcer?tab=readme-ov-file) 和 [*Outlines*](https://github.com/outlines-dev/outlines?tab=readme-ov-file) 在初始化时都需要访问 tokenizer。本教程通过 `inflight_batcher_llm/tensorrt_llm/config.pbtxt` 参数暴露 tokenizer：

```txt
parameters: {
  key: "tokenizer_dir"
  value: {
    string_value: "/Hermes-2-Pro-Llama-3-8B"
  }
}
```
把它直接追加到 `inflight_batcher_llm/tensorrt_llm/config.pbtxt` 的末尾即可。

#### 仓库配置

我们在 [`artifacts/utils.py`](./artifacts/utils.py) 中提供了 *LM Format Enforcer* 和 *Outlines* 的示例实现。请把它复制到 `/opt/tritonserver/inflight_batcher_llm/tensorrt_llm/1/lib`：

```bash
mkdir -p inflight_batcher_llm/tensorrt_llm/1/lib
cp /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/utils.py inflight_batcher_llm/tensorrt_llm/1/lib/
```
最后，安装所有必需的库：

```bash
pip install pydantic lm-format-enforcer outlines setuptools
```

### LM Format Enforcer

要使用 LM Format Enforcer，请确保 `inflight_batcher_llm/tensorrt_llm/1/model.py` 包含以下修改：

```diff
...
import tensorrt_llm.bindings.executor as trtllm

+ from lib.utils import LMFELogitsProcessor, AnswerFormat

...

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    ...

    def get_executor_config(self, model_config):
+       tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
+       logits_processor = LMFELogitsProcessor(tokenizer_dir, AnswerFormat.model_json_schema())
        kwargs = {
            "max_beam_width":
            get_parameter(model_config, "max_beam_width", int),
            "scheduler_config":
            self.get_scheduler_config(model_config),
            "kv_cache_config":
            self.get_kv_cache_config(model_config),
            "enable_chunked_context":
            get_parameter(model_config, "enable_chunked_context", bool),
            "normalize_log_probs":
            get_parameter(model_config, "normalize_log_probs", bool),
            "batching_type":
            convert_batching_type(get_parameter(model_config,
                                                "gpt_model_type")),
            "parallel_config":
            self.get_parallel_config(model_config),
            "peft_cache_config":
            self.get_peft_cache_config(model_config),
            "decoding_config":
            self.get_decoding_config(model_config),
+            "logits_post_processor_map":{
+                LMFELogitsProcessor.PROCESSOR_NAME: logits_processor
+            }
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExecutorConfig(**kwargs)
...
```

#### 发送推理请求

首先，启动 Triton SDK 容器：
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials/:/tutorials \
    -v /path/to/tutorials/repo:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

提供的客户端脚本用到了 `pydantic` 库，SDK 容器中没有自带。继续之前请先安装：

```bash
pip install pydantic
```

##### 方式 1：使用提供的[客户端脚本](./artifacts/client.py)

先发送一个不强制 JSON 回答格式的普通请求：
```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100
```

你应该会收到类似下面的响应：

```bash
Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and
```

现在，在请求中指定 `logits_post_processor_name`：

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100 --logits-post-processor-name "lmfe"
```

这次，预期的响应看起来像：
```bash
Who is Harry Potter?
		{
			"name": "Harry Potter",
			"occupation": "Wizard",
			"house": "Gryffindor",
			"wand": {
				"wood": "Holly",
				"core": "Phoenix feather",
				"length": 11
			},
			"blood_status": "Pure-blood",
			"alive": "Yes"
		}
```
可以看到，[`utils.py`](./artifacts/utils.py) 中定义的 schema 被严格遵守了。注意，LM Format Enforcer 允许 LLM 控制生成字段的顺序，因此字段可以重新排序。

##### 方式 2：使用 [generate 端点](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint)

先发送一个不强制 JSON 回答格式的普通请求：
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```

你应该会收到类似下面的响应：

```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and"}
```

现在，在请求中指定 `logits_post_processor_name`：

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2, "logits_post_processor_name": "lmfe"}'
```

这次，预期的响应看起来像：
```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter?  \t\t\t\n\t\t{\n\t\t\t\"name\": \"Harry Potter\",\n\t\t\t\"occupation\": \"Wizard\",\n\t\t\t\"house\": \"Gryffindor\",\n\t\t\t\"wand\": {\n\t\t\t\t\"wood\": \"Holly\",\n\t\t\t\t\"core\": \"Phoenix feather\",\n\t\t\t\t\"length\": 11\n\t\t\t},\n\t\t\t\"blood_status\": \"Pure-blood\",\n\t\t\t\"alive\": \"Yes\"\n\t\t}\n\n\t\t\n\n\n\n\t\t\n"}
```

### Outlines

要使用 Outlines，请确保 `inflight_batcher_llm/tensorrt_llm/1/model.py` 包含以下修改：

```diff
...
import tensorrt_llm.bindings.executor as trtllm

+ from lib.utils import OutlinesLogitsProcessor, AnswerFormat

...

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    ...

    def get_executor_config(self, model_config):
+       tokenizer_dir = model_config['parameters']['tokenizer_dir']['string_value']
+       logits_processor = OutlinesLogitsProcessor(tokenizer_dir, AnswerFormat.model_json_schema())
        kwargs = {
            "max_beam_width":
            get_parameter(model_config, "max_beam_width", int),
            "scheduler_config":
            self.get_scheduler_config(model_config),
            "kv_cache_config":
            self.get_kv_cache_config(model_config),
            "enable_chunked_context":
            get_parameter(model_config, "enable_chunked_context", bool),
            "normalize_log_probs":
            get_parameter(model_config, "normalize_log_probs", bool),
            "batching_type":
            convert_batching_type(get_parameter(model_config,
                                                "gpt_model_type")),
            "parallel_config":
            self.get_parallel_config(model_config),
            "peft_cache_config":
            self.get_peft_cache_config(model_config),
            "decoding_config":
            self.get_decoding_config(model_config),
+            "logits_post_processor_map":{
+                OutlinesLogitsProcessor.PROCESSOR_NAME: logits_processor
+            }
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExecutorConfig(**kwargs)
...
```

#### 发送推理请求

首先，启动 Triton SDK 容器：
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials/:/tutorials \
    -v /path/to/tutorials/repo:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

提供的客户端脚本用到了 `pydantic` 库，SDK 容器中没有自带。继续之前请先安装：

```bash
pip install pydantic
```

##### 方式 1：使用提供的[客户端脚本](./artifacts/client.py)

先发送一个不强制 JSON 回答格式的普通请求：
```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100
```

你应该会收到类似下面的响应：

```bash
Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and
```

现在，在请求中指定 `logits_post_processor_name`：

```bash
python3 /tutorials/AI_Agents_Guide/Constrained_Decoding/artifacts/client.py --prompt "Who is Harry Potter?" -o 100 --logits-post-processor-name "outlines"
```

这次，预期的响应看起来像：
```bash
Who is Harry Potter?{ "name": "Harry Potter","house": "Gryffindor","blood_status": "Pure-blood","occupation": "Wizards","alive": "No","wand": {"wood": "Holly","core": "Phoenix feather","length": 11 }}
```
可以看到，[`utils.py`](./artifacts/utils.py) 中定义的 schema 被严格遵守了。注意，LM Format Enforcer 允许 LLM 控制生成字段的顺序，因此字段可以重新排序。

##### 方式 2：使用 [generate 端点](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0#query-the-server-with-the-triton-generate-endpoint)

先发送一个不强制 JSON 回答格式的普通请求：
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```

你应该会收到类似下面的响应：

```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter? Harry Potter is a fictional character in a series of fantasy novels written by British author J.K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and"}
```

现在，在请求中指定 `logits_post_processor_name`：

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Who is Harry Potter?", "max_tokens": 100, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2, "logits_post_processor_name": "outlines"}'
```

这次，预期的响应看起来像：
```bash
{"context_logits":0.0,...,"text_output":"Who is Harry Potter?{ \"name\": \"Harry Potter\",\"house\": \"Gryffindor\",\"blood_status\": \"Pure-blood\",\"occupation\": \"Wizards\",\"alive\": \"No\",\"wand\": {\"wood\": \"Holly\",\"core\": \"Phoenix feather\",\"length\": 11 }}"}
```
