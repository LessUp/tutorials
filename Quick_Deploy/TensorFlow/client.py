# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import tritonclient.http as httpclient
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tritonclient.utils import triton_to_np_dtype


# 预处理函数：读取图片并转换为 ResNet50 要求的 224x224 输入
def process_image(image_path="img1.jpg"):
    img = image.load_img(image_path, target_size=(224, 224))
    # 转为数组并增加 batch 维度（(1, 224, 224, 3)）
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 按 ImageNet 的 Keras 约定做预处理（去均值、按通道缩放）
    return preprocess_input(x)


transformed_img = process_image()

# 建立与 Triton 服务器的 HTTP 连接（默认端口 8000）
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# 声明输入输出张量：名称须与 SavedModel 的输入层（input_1）、输出层（predictions）一致
inputs = httpclient.InferInput("input_1", transformed_img.shape, datatype="FP32")
inputs.set_data_from_numpy(transformed_img, binary_data=True)

output = httpclient.InferRequestedOutput(
    "predictions", binary_data=True, class_count=1000
)

# 向服务器发送推理请求并等待结果
results = triton_client.infer(model_name="resnet50", inputs=[inputs], outputs=[output])

predictions = results.as_numpy("predictions")
print(predictions)
