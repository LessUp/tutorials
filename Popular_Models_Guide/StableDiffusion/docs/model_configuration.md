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

# Stable Diffusion 模型配置选项

示例 Python 后端
[`/backend/diffusion/model.py`](../backend/diffusion/model.py)
支持下列配置参数，用于定制所服务的模型。

## 完整配置示例

   * [Stable Diffusion v1.5](../diffusion-models/stable_diffusion_1_5/config.pbtxt)
   * [Stable Diffusion XL](../diffusion-models/stable_diffusion_xl/config.pbtxt)

## 批大小与动态批处理

你可以设置批大小（batch size）和动态批处理（dynamic batching）的排队延迟。批大小为 1 时动态批处理会被禁用。

> [!Note]
> 修改批大小需要重新构建 TensorRT 引擎


```bash
max_batch_size: 1

dynamic_batching {
 max_queue_delay_microseconds: 100000
}

```

> 💡 **AI Infra 视角**：动态批处理（dynamic batching）是 Triton 提升吞吐的核心机制：多个到达时刻不同的请求会在队列里短暂攒批，凑满 `max_batch_size` 或等够 `max_queue_delay_microseconds`（最大排队延迟）后一次性送进引擎执行。批越大，GPU 利用率越高，但攒批会引入额外的排队延迟——这个"攒批时间 vs 批大小"的权衡是调优吞吐时的核心旋钮。需要注意扩散模型这类引擎是按固定批大小编译的，改批大小必须重新构建引擎。

## 引擎构建参数

下面的配置参数会影响引擎构建。

更多信息请参考 [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion)。

```
{
  key: "onnx_opset"
  value: {
    string_value: "18"
  }
},
{
  key: "image_height"
  value: {
    string_value: "512"
  }
},
{
  key: "image_width"
  value: {
    string_value: "512"
  }
},
{
  key: "version"
  value: {
    string_value: "1.5"
  }
}
```

> 💡 **AI Infra 视角**：`onnx_opset`（ONNX 算子集版本）决定导出 ONNX 时使用哪个版本的算子规范；`image_height`/`image_width` 则决定了引擎编译时锁定的图像尺寸。这些参数在构建期就固化进了引擎——意味着部署时若想换分辨率或换导出格式，只能重新构建引擎，而不是改个运行时配置就能生效。设计模型服务 API 时，最好把这类"编译期参数"和"运行时参数"（如 `steps`）明确分开。

## 强制重建引擎

将下面的参数设为非空值会强制重新构建引擎。

```
{
  key: "force_engine_build"
  value: {
    string_value: ""
  }
}
```

## 运行时设置

下面的配置参数影响模型的运行时行为。
更多信息请参考 [TensorRT demo](https://github.com/NVIDIA/TensorRT/tree/release/9.2/demo/Diffusion)。

为 `seed` 设置一个非空的整数值会得到确定性的结果。

```
{
  key: "steps"
  value: {
    string_value: "50"
  }
},
{
  key: "scheduler"
  value: {
    string_value: ""
  }
},
{
  key: "guidance_scale"
  value: {
    string_value: "7.5"
  }
},
{
  key: "seed"
  value: {
    string_value: ""
  }
}
```

> 💡 **AI Infra 视角**：`steps`（去噪步数）和 `guidance_scale`（引导强度）是扩散模型两个最影响成本与效果的参数：步数越多、引导越强，生成质量越好，但推理延迟几乎线性上升。生产服务通常把步数做成用户可选的档位（如快速/标准/精细），并在服务端限制上限，防止个别用户把 GPU 资源吃满。`seed` 置空表示每次随机，固定则可复现结果，方便做回归测试和效果对比。
