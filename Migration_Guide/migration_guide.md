# 迁移到 Triton Inference Server

迁移到新的推理技术栈听起来可能有些挑战性，但只要把难点拆解开、理解其中的最佳实践，这件事就没那么难。本指南会展示使用 Triton Inference Server 这类专用推理服务方案带来的好处，并帮你判断哪条迁移路径最适合你的情况。

## 为什么需要专用的推理解决方案？

搭建推理服务所需的基础设施可能相当复杂。我们先考虑一个最简单的场景：没有扩缩容、单节点部署、不需要负载均衡器。这种情况下，要提供一个模型服务需要什么？

如果你用 Python 开发，刚接触模型推理领域，或者只是想快速搭个东西出来，你可能会想到 [Flask](https://flask.palletsprojects.com/en/2.2.x/) 这类工具：一个灵活的微框架，可以按需扩展生态。在 Flask 里提供服务，只需要一个能处理 POST 请求的函数。

```python
@app.route('/run_my_model',methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Pre-processing
    ...

    prediction = model(<data>)

    # Post-processing
    ...

    return output
```

几行代码，模型就跑起来了。任何人都能发送请求并使用这个模型！但是等等，当请求量开始增多时会发生什么？我们需要一种方式把这些任务/请求排队。比如说，用 [Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html) 来解决排队问题，顺手还能加一个响应缓存来应对重复查询。

![Flask 流程图](./img/arch.PNG)

上面的方案确实能用，但限制很多，资源利用率也低。为什么这么说？假设我们有一个图像分类模型，最大批大小（batch size）为 `64`，服务器每 `100 ms` 收到 `50` 个请求。如果不实现任何批处理策略，这些请求只能串行处理，GPU 资源被白白浪费。而这只是冰山一角。再看看下面这些场景：
* 如果要支持多个模型，每次更新模型都要重启服务器吗？
* 模型如何做版本管理？
* 同一个服务器上能同时跑 PyTorch 和 TensorFlow 模型吗？
* 如果一个模型需要 CPU、另一个需要 GPU，怎么优化执行？
* 同一个节点上的多张 GPU 怎么管理？
* 执行时间是否已优化？I/O 处理是否高效？
* 模型集成（model ensemble）怎么处理？
* 监控服务器指标的最佳方式是什么？

这些只是我们需要考虑并投入工程时间解决的问题中的一小部分。而且这些功能还需要针对软件、硬件加速器或执行环境的每个大版本定期维护和优化。随着部署规模扩大，这些挑战只会越来越严峻。显然，解决方案不能是让每个开发者都从通用框架起步，自建并维护一套基础设施。这正是 Triton Inference Server 这类专用推理服务器大显身手的地方。

> 💡 **AI Infra 视角**：Flask + Celery 这类 DIY 方案能跑通，但缺少推理服务真正核心的能力：动态批处理（dynamic batching）、并发请求调度、GPU 显存复用。Triton 这类专用推理服务器把"请求排队 → 攒批 → 调度到 GPU"这条链路做成了开箱即用的能力，吞吐和 GPU 利用率通常能提升一个数量级。

## 如何把工作流迁移到 Triton？

本指南假设你对 Triton Inference Server 的基础概念有所了解。如果你还不熟悉它，建议先看这个[入门视频](https://www.youtube.com/watch?v=NQDtfSi5QF4)和这篇[概念指南](../Conceptual_Guide/Part_1-model_deployment/README.md)。

每条现有推理管线都有各自的形态，因此不存在"放之四海而皆准"的迁移方案。不过，本指南会试着帮你建立迁移过程的直觉。总体来看，大多数推理栈可以归入四大类。

> 💡 **AI Infra 视角**：与 TF Serving、ONNX Runtime 等其它推理服务框架相比，Triton 最大的差异化在于"后端插件化"：PyTorch、TensorFlow、ONNX 乃至自定义后端可以共存于同一个服务进程，由统一的前端（HTTP/gRPC）对外暴露。迁移时你不需要"换框架"，而是把模型放进去、对齐请求格式，真正的复杂度在于模型配置（model config）和模型仓库（model repository）的组织方式。

* **与更大的模块强耦合**：也许你正在迭代或微调一个模型，把模型从现有技术栈中解耦出来需要相当大的投入。你仍然需要更好的性能来尽可能少占硬件资源，并和内部团队共享开发分支。隔离依赖、导出模型、搭建某种模型存储……这些工作的成本都不现实。你需要的是一个能注入现有代码库、既不侵入也不耗时的新方案。

    这种情况下，我们推荐使用 [PyTriton](https://github.com/triton-inference-server/pytriton)，它是一个类似 Flask/FastAPI 的接口，用户可以通过它借助 Triton Inference Server 为业务场景提供服务。

    ```python
    from pytriton.decorators import sample
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton

    MODEL = ...

    @sample
    def <your_function_name>(sequence: np.ndarray, labels: np.ndarray):
        # Decode input
        sequence = np.char.decode(sequence.astype("bytes"), "utf-8")
        labels = np.char.decode(labels.astype("bytes"), "utf-8")

        result = MODEL(...)

        return {"scores": results}

    # PyTriton code
    with Triton() as triton:
        triton.bind(
            model_name="<model name>",
            infer_func=<your_function_name>,      # function you want to serve
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(1,)),
                Tensor(name="labels", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="scores", dtype=np.float32, shape=(-1,)),
            ],
            # add the features you want to enable here
            config=ModelConfig(batching=False),
        )
        triton.serve()
    ```

    上面的例子是[这个示例](https://github.com/triton-inference-server/pytriton/tree/main/examples/huggingface_bart_pytorch)的骨架版本。要点是：任何你想提供的函数——无论是包含模型推理的部分，还是纯 Python 代码——都可以绑定到 Triton 上。作为用户，你完全不用操心如何启动 Triton Inference Server 或搭建模型仓库（model repository），这些步骤全部由 PyTriton 库代为处理。更多架构细节可以[在这里](https://triton-inference-server.github.io/pytriton/latest/high_level_design)找到。

* **松耦合，但管线盘根错节**：假设你要提供的管线可以隔离到独立环境中。通常到这个阶段，模型和管线已经过内部测试且结果令人满意。但管线可能仍然纠缠不清：部分模型无法导出，前后处理步骤与管线逻辑依然紧密耦合。

    这种情况下，用户仍然可以使用 PyTriton；但如果部分模型可以导出，改用 Triton 的 Python 后端（Python Backend）搭配其他框架后端，可以获得更高的性能。概念指南的[第 6 部分](../Conceptual_Guide/Part_6-building_complex_pipelines/README.md)就是这类场景的绝佳示例。

    目前 PyTriton 还无法覆盖 Triton Inference Server 的全部特性。想用上完整功能集的用户，也可以选择 Python 后端。这个 [HuggingFace 示例](../HuggingFace/README.md#deploying-on-the-python-backend-approach-1)会带你了解具体细节。

* **松耦合、模块化的管线**：随着管线复杂度上升，深度学习管线之间经常出现大量重叠——多个管线共用同一组模型或前后处理步骤。这种情况下，把管线的所有组件都部署到 Triton Inference Server 上，再[构建模型集成（model ensemble）](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)，收益极大。即便没有重叠，用 Triton 的模型集成来管理管线，也能获得扩展性和性能上的好处。深入讲解请参考[这份指南](../Conceptual_Guide/Part_5-Model_Ensembles/README.md)。

* **只部署模型，不带前后处理**：很多情况下，管线逻辑以经过多年打磨的高度优化底层脚本的形式存在。此时用户可能只想部署模型本身，避免 HTTP/gRPC 网络调用——因为模型是嵌入在更大的应用里被消费的。这种情况下，可以通过 Triton 的[共享内存扩展](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md#shared-memory-extension)和 [C API](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inprocess_c_api.md)访问模型，完全省去网络接口。

> 💡 **AI Infra 视角**：无论选哪条迁移路径，最终落点都是 Triton 的模型仓库（model repository）结构：`<模型名>/<版本号>/` 目录 + `config.pbtxt` 配置文件。这份配置里声明的输入输出形状、`max_batch_size`、动态批处理、`instance_group` 等参数，直接决定了推理服务能榨出多少硬件性能——迁移时最值得花时间吃透的就是它。

## 结论

模型的部署方式多种多样，每种情况都有自己的挑战和要求。借助 Triton Inference Server 的各种特性，这些需求都能得到满足。我们鼓励你深入研究 Triton Inference Server 的[文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)，了解更多特性细节！
