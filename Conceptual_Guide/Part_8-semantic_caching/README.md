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

# 语义缓存（Semantic Caching）

在部署大语言模型（LLM）或基于 LLM 的工作流时，有两个关键因素需要考虑：应用的性能和成本效率。生成语言模型输出需要大量的计算资源，例如 GPU 时间、内存占用以及其他基础设施成本。这些资源密集型的需求催生了优化策略的迫切需求——在保持高质量输出的同时尽量降低运营开销。

语义缓存正是降低 LLM 应用计算成本的一个有力方案。

## 定义与收益

**_语义缓存（Semantic caching）_** 是一种把传入请求的语义（semantics）纳入考量的缓存机制，而不仅仅是原始数据本身。它超越了简单的键值对，会考虑数据的内容或上下文。

这种方法带来的收益包括（但不限于）：

+ **成本优化（Cost Optimization）**

    - 语义缓存可以大幅降低 LLM 部署相关的运营开销。通过存储并复用语义相似查询的响应，它把实际需要调用的 LLM 次数降到最低。

+ **降低延迟（Reduced Latency）**

    - 语义缓存的主要收益之一是能显著改善响应时间。通过为相似查询直接取回缓存响应，系统可以绕过完整的模型推理，从而降低延迟。

+ **提高吞吐（Increased Throughput）**

    - 语义缓存让计算资源得到更高效的利用。通过为相似查询提供缓存响应，它减轻了基础设施组件的负载。这种高效性使系统能用同样的硬件处理更大量的请求，实际上提高了吞吐。

+ **可扩展性（Scalability）**

    - 随着用户规模和查询量的增长，只要有足够的存储和资源支持这种扩展，缓存命中率就会随之提高。资源效率的提升和计算需求的降低，让应用可以在不按比例增加基础设施成本的情况下服务更多用户。

+ **响应一致性（Consistency in Responses）**

    - 对于某些应用，保持相似查询响应的一致性是有益的。语义缓存确保类似的问题得到统一的答案，这在客户服务或教育类场景中尤其有用。

> 💡 **AI Infra 视角**：LLM 推理的成本大头是 GPU 时间，而实际线上流量里大量查询是重复或近似的（客服问答、文档检索、代码补全提示）。语义缓存的价值在于用"一次向量检索 + 内存读取"（毫秒级）替代一次完整生成（百毫秒到秒级），相当于免费获得 N 倍吞吐。设计要点：相似度阈值要按业务容忍度调，且必须考虑 prompt 参数（temperature、max_tokens）是否参与缓存键——本教程末尾的局限部分就指出了这一点。

## 参考实现示例（Sample Reference Implementation）

本教程在 [semantic_caching.py](./artifacts/semantic_caching.py) 中提供了一个语义缓存的参考实现。它有三个关键依赖：

* [SentenceTransformer](https://sbert.net/)：一个用于计算句子、段落和图片的稠密向量表示（embeddings）的 Python 框架。
    - 我们用这个库（特别是 `all-MiniLM-L6-v2`）把传入的 prompt 转换为 embedding，从而实现语义比较。
    - 备选方案包括[语义搜索模型](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models)、OpenAI Embeddings 等。
* [Faiss](https://github.com/facebookresearch/faiss/wiki)：Facebook AI Research 开发的开源库，用于稠密向量的高效相似度搜索和聚类。
    - 这个库用于 embedding 存储，以及从缓存的请求（或索引存储）中提取最相似的已嵌入 prompt。
    - 这是一个功能强大的库，提供各种各样的 CPU 和 GPU 加速算法。
    - 备选方案包括 [annoy](https://github.com/spotify/annoy) 或 [cuVS](https://github.com/rapidsai/cuvs)。不过注意，cuVS 已经集成在 Faiss 中，更多信息见[这里](https://docs.rapids.ai/api/cuvs/nightly/integrations/faiss/)。
* [Theine](https://github.com/Yiling-J/theine)：高性能的内存缓存。
    - 我们将把它用作精确匹配缓存的后端。找到最相似的 prompt 后，从缓存中取出对应的缓存响应。这个库支持多种淘汰策略，本教程使用 "LRU"。
    - 也可以考虑 [MemCached](https://memcached.org/about) 作为备选方案。

提供的[脚本](./artifacts/semantic_caching.py)注释非常详尽，我们鼓励用户通读代码，以便更清楚地理解各个必要阶段。

## 把语义缓存集成到你的工作流

本教程以 [vllm backend](https://github.com/triton-inference-server/vllm_backend) 为例，重点演示如何为非流式（non-streaming）场景缓存响应。这里讲的原则也可以推广到流式场景。

### 定制 vLLM Backend

首先，克隆 Triton 的 vllm backend 仓库。这将提供实现我们语义缓存示例所需的代码库。

```bash
git clone https://github.com/triton-inference-server/vllm_backend.git
cd vllm_backend
```

仓库克隆成功后，下一步是应用所有必要的修改。为了简化这个过程，我们准备了一个 [semantic_cache.patch](./artifacts/semantic_cache.patch)，把全部改动合并成一步：

```bash
curl https://raw.githubusercontent.com/triton-inference-server/tutorials/refs/heads/main/Conceptual_Guide/Part_8-semantic_caching/artifacts/semantic_cache.patch | git apply -v
```

如果你急着开始使用优化后的 vLLM backend 跑 Triton，可以直接跳到[使用优化后的 vLLM Backend 启动 Triton](#launching-triton-with-optimized-vllm-backend)一节。不过，如果你对细节感兴趣，我们来看看这个 patch 包含什么。

这个 patch 引入了一个新脚本 [semantic_caching.py](./artifacts/semantic_caching.py)，并把它放入合适的目录。这个脚本实现了语义缓存功能的核心逻辑。

接下来，patch 把语义缓存集成到模型中。我们一步步来看这些改动。

首先，它把 [semantic_caching.py](./artifacts/semantic_caching.py) 中的必要类导入代码库：

```diff
...

from utils.metrics import VllmStatLogger
+from utils.semantic_caching import SemanticCPUCacheConfig, SemanticCPUCache
```

接下来，它在初始化阶段设置语义缓存。这个设置会让你的模型在运行期间使用语义缓存。

```diff
    def initialize(self, args):
        self.args = args
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        ...

        # Starting asyncio event loop to process the received requests asynchronously.
        self._loop = asyncio.get_event_loop()
        self._event_thread = threading.Thread(
            target=self.engine_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._event_thread.start()
+       config = SemanticCPUCacheConfig()
+       self.semantic_cache = SemanticCPUCache(config=config)

```

最后，patch 在请求处理过程中加入了查询和更新语义缓存的逻辑。这确保缓存响应在可能的情况下被高效利用。

```diff
    async def generate(self, request):
        ...
        try:
            request_id = random_uuid()
            prompt = pb_utils.get_input_tensor_by_name(
                request, "text_input"
            ).as_numpy()[0]
            ...

            if prepend_input and stream:
                raise ValueError(
                    "When streaming, `exclude_input_in_output` = False is not allowed."
                )
+           cache_hit = self.semantic_cache.get(prompt)
+           if cache_hit:
+               try:
+                   response_sender.send(
+                   self.create_response(cache_hit, prepend_input),
+                   flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
+                   )
+                   if decrement_ongoing_request_count:
+                       self.ongoing_request_count -= 1
+               except Exception as err:
+                   print(f"Unexpected {err=} for prompt {prompt}")
+               return None
            ...

            async for output in response_iterator:
                ...

            last_output = output

            if not stream:
                response_sender.send(
                    self.create_response(last_output, prepend_input),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
+               self.semantic_cache.set(prompt, last_output)

```

### 使用优化后的 vLLM Backend 启动 Triton

为了评估优化后的 vllm backend，我们启动 vllm docker 容器，并把我们的实现挂载到 `/opt/tritonserver/backends/vllm`。我们还会挂载示例模型仓库，它位于 `vllm_backend/samples/model_repository`。你也可以自由设置自己的仓库。用下面的 docker 命令启动 Triton 的 vllm docker 容器，但请确保把路径换成克隆的 `vllm_backend` 仓库的正确路径，并把 `<xx.yy>` 替换为 Triton 的最新版本。

```bash
docker run --gpus all -it --net=host --rm \
    --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/vllm_backend/src/:/opt/tritonserver/backends/vllm \
    -v /path/to/vllm_backend/samples/model_repository:/workspace/model_repository \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3
```

进入容器后，确保安装所需依赖：

```bash
pip install sentence_transformers faiss_gpu theine
```

最后启动 Triton：

```bash
tritonserver --model-repository=model_repository/
```

启动 Triton 后，你会在控制台上看到服务器启动和加载模型的输出。当你看到类似下面的输出时，Triton 就可以接受推理请求了。

```
I1030 22:33:28.291908 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1030 22:33:28.292879 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1030 22:33:28.335154 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

### 评估

用示例 model_repository [启动 Triton](#launching-triton-with-optimized-vllm-backend) 后，你可以用 [generate 端点](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md)快速发起你的第一个推理请求。

我们还会给这个查询计时：

```bash
time curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "Tell me, how do I create model repository for Triton Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens":100}, "exclude_input_in_output":true}'
```

成功后，你应该看到服务器返回类似这样的响应：

```
{"model_name":"vllm_model","model_version":"1","text_output": <MODEL'S RESPONSE>}
real	0m1.128s
user	0m0.000s
sys	0m0.015s
```

现在，我们换一种说法，但保持语义相同：

```bash
time curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "How do I set up model repository for Triton Inference Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens":100}, "exclude_input_in_output":true}'
```

成功后，你应该看到服务器返回类似这样的响应：

```
{"model_name":"vllm_model","model_version":"1","text_output": <SAME MODEL'S RESPONSE>}
real	0m0.038s
user	0m0.000s
sys	0m0.017s
```

再试一次：

```bash
time curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "How model repository should be set up for Triton Server?", "parameters": {"stream": false, "temperature": 0, "max_tokens":100}, "exclude_input_in_output":true}'
```

成功后，你应该看到服务器返回类似这样的响应：

```
{"model_name":"vllm_model","model_version":"1","text_output": <SAME MODEL'S RESPONSE>}
real	0m0.059s
user	0m0.016s
sys	0m0.000s
```

很明显，后两个请求与第一个请求在语义上相似，这触发了缓存命中场景，把模型的延迟从约 1.1 秒降到了平均每请求 0.048 秒。

## 当前的局限（Current Limitations）

* 当前语义缓存的实现只考虑 prompt 本身来决定缓存命中，没有考虑 `max_tokens`、`temperature` 等附加请求参数。因此这些参数不参与缓存命中评估，在使用不同配置时可能会影响缓存响应的准确性。

* 语义缓存的有效性高度依赖 embedding 模型的选择和应用场景。例如，"How to set up model repository for Triton Inference Server?" 和 "How not to set up model repository for Triton Inference Server?" 这两条查询可能有很高的余弦相似度，但语义截然不同。这让设置最优的缓存命中阈值变得很有挑战性，因为相似度范围收窄可能会排除有用的缓存条目。

## 对这个特性感兴趣吗？

这个参考实现让你一窥语义缓存的潜力，但请注意，它并不是 Triton Inference Server 官方支持的特性。

我们很重视你的意见！如果你希望语义缓存在未来版本中成为受支持的特性，欢迎加入正在进行的[讨论](https://github.com/triton-inference-server/server/discussions/7742)。请说明为什么你认为语义缓存对你的用例有价值。你的反馈会帮助塑造我们的产品路线图，我们感谢你为让我们的软件对每个人更好所做的贡献。
