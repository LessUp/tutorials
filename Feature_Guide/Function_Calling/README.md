<!--
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
-->

# Triton Inference Server 中的函数调用（Function Calling）

本教程聚焦函数调用（function calling）——一种将大语言模型（LLM）轻松接入外部工具的常见方法。这种方式赋予 AI 智能体（agent）有效的工具使用能力，使其能与外部 API 无缝交互，显著拓展了它们的能力和实际应用范围。

## 目录

- [什么是函数调用？](#什么是函数调用)
- [教程概览](#教程概览)
    + [前置条件：Hermes-2-Pro-Llama-3-8B](#前置条件hermes-2-pro-llama-3-8b)
- [函数定义](#函数定义)
- [提示工程](#提示工程)
- [整合在一起](#整合在一起)
- [进一步优化](#进一步优化)
    + [强制输出格式](#强制输出格式)
    + [并行工具调用](#并行工具调用)
- [参考资料](#参考资料)

## 什么是函数调用？

函数调用指的是 LLM 具备以下能力：
 * 识别何时需要使用某个特定函数或工具来回答问题或执行任务。
 * 生成包含调用该函数所需参数的结构化输出。
 * 把函数调用的结果整合进自己的回答中。

函数调用是一种强大的机制，它让 LLM 能够执行更复杂的任务（例如多智能体系统中的智能体编排），这些任务需要超出模型固有知识范围的具体计算或数据检索。通过识别何时需要某个特定函数，LLM 可以动态扩展自身功能，在真实应用中变得更加通用和实用。

> 💡 **AI Infra 视角**：函数调用（function calling）是 Agent 应用的事实标准交互范式：模型输出的是「调用哪个工具 + 参数 JSON」的结构化声明，而不是直接执行代码——真正的执行发生在服务端或客户端编排层。对推理服务来说，函数调用并不改变模型推理本身，重点是三个工程问题：把工具 schema 塞进 prompt 的 token 成本、保证输出能被可靠解析成合法 JSON、以及多轮「模型↔工具」往返之间的会话状态管理。

## 教程概览

本教程使用 [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 模型演示函数调用，该模型针对这一能力做过预微调。我们将创建一个基础的股票播报智能体，它提供最新的股票信息并总结近期公司新闻。

### 前置条件：Hermes-2-Pro-Llama-3-8B

在继续之前，请确保你已经按照[这些步骤](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)用 Triton Inference Server 和 TensorRT-LLM backend 成功部署了 [Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 模型。

> [!IMPORTANT]
> 启动 docker 容器时，确保 `tutorials` 文件夹挂载到了 `/tutorials`。

## 函数定义

我们为股票播报智能体定义三个函数：
1. `get_current_stock_price`：获取给定股票代码的当前股价。
2. `get_company_news`：获取给定股票代码的公司新闻和新闻稿。
3. `final_answer`：作为空操作使用，用于指示最终回答。

每个函数都包含名称、描述和输入参数 schema：
 ```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_stock_price",
            "description": "Get the current stock price for a given symbol.\n\nArgs:\n  symbol (str): The stock symbol.\n\nReturns:\n  float: The current stock price, or None if an error occurs.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get company news and press releases for a given stock symbol.\n\nArgs:\nsymbol (str): The stock symbol.\n\nReturns:\npd.DataFrame: DataFrame containing company news and press releases.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return final generated answer",
            "parameters": {
                "type": "object",
                "properties": {"final_response": {"type": "string"}},
                "required": ["final_response"],
            },
        },
    },
]
 ```
这些函数定义将通过提示（prompt）传给模型，使它在对话过程中能识别并恰当地使用这些函数。

函数的具体实现请参考 [client_utils.py](./artifacts/client_utils.py)。

> 💡 **AI Infra 视角**：工具定义的写法直接影响调用成功率：`description` 是模型路由到哪个工具的主要依据，要写清楚「什么时候用、参数怎么填」；`parameters` 用 JSON Schema 描述参数类型和必填项。对服务端来说，工具列表会整体拼进系统提示词——工具越多，prefill 阶段的输入越长，TTFT 和显存占用都会上升，所以生产环境通常按会话按需注入工具子集，而不是把全部工具塞给每个请求。

## 提示工程

**提示工程（prompt engineering）** 是函数调用的关键环节，它引导 LLM 识别何时以及如何使用特定函数。通过精心构造提示，你可以有效定义 LLM 的角色、目标和可访问的工具，确保任务执行准确高效。

针对本任务，我们组织了一个示例提示结构，放在随附的 [`system_prompt_schema.yml`](./artifacts/system_prompt_schema.yml) 文件中。该文件细致地列出了：

- **角色（Role）**：定义 LLM 预期扮演的具体角色。
- **目标（Objective）**：明确说明交互的目标或期望结果。
- **工具（Tools）**：列出 LLM 可用的函数或工具，用于实现目标。
- **Schema**：指定调用每个工具或函数所需的结构和格式。
- **指令（Instructions）**：提供一套清晰的准则，确保 LLM 沿着预期路径前进并恰当使用工具。

通过利用提示工程，你可以增强 LLM 执行复杂任务的能力，把函数调用无缝整合进它的回答中，从而在各类应用中最大化它的价值。

## 整合在一起

首先，启动 Triton SDK 容器：
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials/:/tutorials \
    -v /path/to/tutorials/repo:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

提供的客户端脚本用到了 `pydantic` 和 `yfinance` 库，SDK 容器中没有自带。继续之前请先安装：

```bash
pip install pydantic yfinance
```

按如下方式运行提供的 [`client.py`](./artifacts/client.py)：

```bash
python3 /tutorials/AI_Agents_Guide/Function_Calling/artifacts/client.py --prompt "Tell me about Rivian. Include current stock price in your final response." -o 200
```

你应该会看到类似下面的响应：

```bash
+++++++++++++++++++++++++++++++++++++
RESPONSE: Rivian, with its current stock price of <CURRENT STOCK PRICE>, <NEWS SUMMARY>
+++++++++++++++++++++++++++++++++++++
```

要查看 LLM "调用"了哪些工具，只需加上 `verbose` 标志：

```bash
python3 /tutorials/AI_Agents_Guide/Function_Calling/artifacts/client.py --prompt "Tell me about Rivian. Include current stock price in your final response." -o 200 --verbose
```

这会展示函数调用的逐步过程，包括：
- 被调用的工具
- 传给每个工具的参数
- 每次函数调用的响应
- 最终的总结性回答


```bash
[b'\n{\n  "step": "1",\n  "description": "Get the current stock price for Rivian",\n  "tool": "get_current_stock_price",\n  "arguments": {\n    "symbol": "RIVN"\n  }\n}']
=====================================
Executing function: get_current_stock_price({'symbol': 'RIVN'})
Function response: <CURRENT STOCK PRICE>
=====================================
[b'\n{\n  "step": "2",\n  "description": "Get company news and press releases for Rivian",\n  "tool": "get_company_news",\n  "arguments": {\n    "symbol": "RIVN"\n  }\n}']
=====================================
Executing function: get_company_news({'symbol': 'RIVN'})
Function response: [<LIST OF RECENT NEWS TITLES>]
=====================================
[b'\n{\n  "step": "3",\n  "description": "Summarize the company news and press releases for Rivian",\n  "tool": "final_answer",\n  "arguments": {\n    "final_response": "Rivian, with its current stock price of  <CURRENT STOCK PRICE>, <NEWS SUMMARY>"\n  }\n}']


+++++++++++++++++++++++++++++++++++++
RESPONSE: Rivian, with its current stock price of  <CURRENT STOCK PRICE>, <NEWS SUMMARY>
+++++++++++++++++++++++++++++++++++++
```

> [!TIP]
> 本教程中，所有功能（工具定义、实现和执行）都在客户端实现（见 [client.py](./artifacts/client.py)）。
> 在生产场景中，尤其是函数预先已知的情况下，建议把这套逻辑放到服务端实现。
> 服务端实现的一种推荐做法是，通过 Triton [ensemble](https://github.com/triton-inference-server/server/blob/a6fff975a214ff00221790dd0a5521fb05ce3ac9/docs/user_guide/architecture.md#ensemble-models) 或 [BLS](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#business-logic-scripting) 部署你的工作流。
> 用一个预处理（pre-processing）模型把用户提示与系统提示、可用工具组合格式化；再用一个后处理（post-processing）模型管理对已部署 LLM 的多次调用，直到得出最终答案。

> 💡 **AI Infra 视角**：客户端实现函数调用适合原型验证，但生产环境通常把它挪到服务端：用 ensemble / BLS 把「提示组装 → LLM 调用 → 工具执行 → 结果回填」编排成一条管线。好处是客户端只需发一次请求、网络往返最少，且多轮工具调用都在服务端完成，避免把中间状态暴露给每个客户端；代价是服务端要管理会话状态和多轮调用的并发控制。吞吐敏感的场景还要注意工具执行（如查 API）的耗时——它是串行插入在两次 LLM 调用之间的，会直接拉长端到端延迟。

## 进一步优化

### 强制输出格式

本教程演示了如何用提示工程强制特定输出格式。期望的结构如下：
```python
  {
    "step" : <Step number>
    "description": <Description of what the step does and its output>
    "tool": <Tool to use>,
    "arguments": {
        <Parameters to pass to the tool as a valid dict>
    }
  }
```
但有些情况下输出可能会偏离这个要求的 schema。例如，考虑下面的提示执行：

```bash
python3 /tutorials/AI_Agents_Guide/Function_Calling/artifacts/client.py --prompt "How Rivian is doing?" -o 500 --verbose
```
这次执行可能会因为无效的 JSON 格式而失败。verbose 输出会显示，LLM 最终的回答是纯文本，而不是预期的 JSON 格式：
```
{
  "step": "3",
  "description": <Description of what the step does and its output>
  "tool": "final_answer",
  "arguments": {
    "final_response": <Final Response>
  }
}
```
幸运的是，这种行为可以用约束解码（constrained decoding）来控制——这项技术引导模型生成满足特定格式和内容要求的输出。我们强烈建议你阅读专门的[约束解码教程](../Constrained_Decoding/README.md)，深入了解并增强管理模型输出的能力。

> [!TIP]
> 为了获得最佳效果，把 [client_utils.py](./artifacts/client_utils.py) 中定义的 `FunctionCall` 类作为 Logits Post-Processor 的 JSON schema 使用。这样可以保证输出一致且格式正确，与本教程中建立的格式结构保持一致。

### 并行工具调用

本教程聚焦单轮强制调用——LLM 在单次交互中被提示执行一个特定函数调用。当需要立刻执行某个精确动作时，这种方法很有用，能确保函数在当前对话中执行。

有些函数调用其实可以同时执行。这种技术对可以拆分成独立操作的任务很有益，能提高效率、缩短响应时间。

我们鼓励读者把实现并行工具调用当作一个练习来挑战自己。

## 参考资料

本教程的部分内容基于 [Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)。
