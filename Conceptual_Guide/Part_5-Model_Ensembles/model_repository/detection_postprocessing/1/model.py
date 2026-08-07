# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import io
import json
import math

import cv2
import numpy as np

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


# 检测后处理模型：把 EAST 的分数/几何输出解码成边界框并裁剪出文字区域，
# 作为文本识别模型的输入（原 Part 1 客户端逻辑，现在在服务端执行）
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    # 模型加载时调用一次（可选）：解析 model_config，准备输出数据类型
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # 必须解析 model_config（JSON 字符串，需要手动解析）
        model_config = json.loads(args["model_config"])

        # 取出输出张量的配置
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "detection_postprocessing_output"
        )

        # 把 Triton 数据类型转换成 numpy 类型
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
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

        output0_dtype = self.output0_dtype

        responses = []

        # 根据旋转矩形的 4 个顶点做透视变换，把文字区域矫正为 (100, 32) 的规整图
        def fourPointsTransform(frame, vertices):
            vertices = np.asarray(vertices)
            outputSize = (100, 32)
            targetVertices = np.array(
                [
                    [0, outputSize[1] - 1],
                    [0, 0],
                    [outputSize[0] - 1, 0],
                    [outputSize[0] - 1, outputSize[1] - 1],
                ],
                dtype="float32",
            )

            rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
            result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
            return result

        # 把 EAST 输出的每个像素的分数与几何信息解码为文字边界框
        def decodeBoundingBoxes(scores, geometry, scoreThresh=0.5):
            detections = []
            confidences = []

            ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
            # 校验输入张量的维度形状是否符合 EAST 输出约定
            assert len(scores.shape) == 4, "Incorrect dimensions of scores"
            assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
            assert scores.shape[0] == 1, "Invalid dimensions of scores"
            assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
            assert scores.shape[1] == 1, "Invalid dimensions of scores"
            assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
            assert (
                scores.shape[2] == geometry.shape[2]
            ), "Invalid dimensions of scores and geometry"
            assert (
                scores.shape[3] == geometry.shape[3]
            ), "Invalid dimensions of scores and geometry"
            height = scores.shape[2]
            width = scores.shape[3]
            for y in range(0, height):
                # 逐像素取出分数与文本框几何参数（四边距离 + 旋转角）
                scoresData = scores[0][0][y]
                x0_data = geometry[0][0][y]
                x1_data = geometry[0][1][y]
                x2_data = geometry[0][2][y]
                x3_data = geometry[0][3][y]
                anglesData = geometry[0][4][y]
                for x in range(0, width):
                    score = scoresData[x]

                    # 分数低于阈值则跳过，不生成候选框
                    if score < scoreThresh:
                        continue

                    # 计算偏移：EAST 特征图相对原图是 4 倍下采样，坐标需乘 4 还原
                    offsetX = x * 4.0
                    offsetY = y * 4.0
                    angle = anglesData[x]

                    # Calculate cos and sin of angle
                    cosA = math.cos(angle)
                    sinA = math.sin(angle)
                    h = x0_data[x] + x2_data[x]
                    w = x1_data[x] + x3_data[x]

                    # Calculate offset
                    offset = [
                        offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                        offsetY - sinA * x1_data[x] + cosA * x2_data[x],
                    ]

                    # Find points for rectangle
                    p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                    p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                    center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                    detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                    confidences.append(float(score))

            # Return detections and confidences
            return [detections, confidences]

        # 遍历所有请求，为每条请求构造一个响应
        for request in requests:
            # 取出三个输入张量：分数、几何信息、预处理后的原图
            in_1 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input_1"
            )
            in_2 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input_2"
            )
            in_3 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input_3"
            )

            # 调整维度顺序后解码边界框，再做 NMS 去掉重叠的检测框
            scores = in_1.as_numpy().transpose(0, 3, 1, 2)
            geometry = in_2.as_numpy().transpose(0, 3, 1, 2)
            frame = np.squeeze(in_3.as_numpy(), axis=0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            [boxes, confidences] = decodeBoundingBoxes(scores, geometry)
            indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

            cropped_list = []
            cv2.imwrite("frame.png", frame)
            count = 0
            for i in indices:
                # 取旋转矩形的 4 个角点，矫正为规整图并转灰度，归一化到 [-1, 1]
                count += 1
                vertices = cv2.boxPoints(boxes[i])
                cropped = fourPointsTransform(frame, vertices)
                cv2.imwrite(str(count) + ".png", cropped)
                cropped = np.expand_dims(
                    cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), axis=0
                )

                cropped_list.append(((cropped / 255.0) - 0.5) * 2)
            cropped_arr = np.stack(cropped_list, axis=0)

            np.save("tensor.pkl", cropped_arr)
            out_tensor_0 = pb_utils.Tensor(
                "detection_postprocessing_output", cropped_arr.astype(output0_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
