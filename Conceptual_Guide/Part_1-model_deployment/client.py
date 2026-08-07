# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import cv2
import numpy as np
import tritonclient.http as httpclient

# 是否保存中间结果图片（调试用）
SAVE_INTERMEDIATE_IMAGES = False


# 文本检测前处理：把原始图片缩放/归一化，转成 EAST 模型需要的 blob 格式
def detection_preprocessing(image: cv2.Mat) -> np.ndarray:
    inpWidth = 640
    inpHeight = 480

    # 前处理：缩放、减均值（123.68, 116.78, 103.94）并归一化
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False
    )
    # 调整通道顺序为 N,H,W,C，与模型输入 input_images:0 的维度约定保持一致
    blob = np.transpose(blob, (0, 2, 3, 1))
    return blob


# 文本检测后处理：把模型的分数与几何输出解码成边界框，并裁剪出文字区域图片
def detection_postprocessing(scores, geometry, preprocessed_image):
    # 根据四个顶点做透视变换，把旋转的文字区域矫正为 (100, 32) 的规整图片
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

    # 解码 EAST 模型的输出：把每个像素位置的分数与几何信息还原为文字边界框
    def decodeBoundingBoxes(scores, geometry, scoreThresh=0.5):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ########
        # 校验 scores / geometry 的维度是否符合 EAST 输出的形状约定
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
            # 逐像素遍历：取出分数和该像素处文本框的几何参数（四边距离 + 旋转角）
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # 分数低于阈值则跳过该像素，不再生成候选框
                if score < scoreThresh:
                    continue

                # 计算偏移：EAST 输出特征图相对原图是 4 倍下采样，需乘 4 还原坐标
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

    # 调整张量维度顺序，转换为解码函数期望的 (1, 特征, H, W) 布局
    scores = scores.transpose(0, 3, 1, 2)
    geometry = geometry.transpose(0, 3, 1, 2)
    frame = np.squeeze(preprocessed_image, axis=0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    [boxes, confidences] = decodeBoundingBoxes(scores, geometry)
    # 对旋转矩形框做 NMS（非极大值抑制），去掉重叠的重复检测框
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

    cropped_list = []
    cv2.imwrite("frame.png", frame)
    count = 0
    for i in indices:
        # 取旋转矩形的 4 个角点，矫正为规整图并转灰度
        count += 1
        vertices = cv2.boxPoints(boxes[i])
        cropped = fourPointsTransform(frame, vertices)
        cv2.imwrite(str(count) + ".png", cropped)
        cropped = np.expand_dims(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), axis=0)

        # 归一化到 [-1, 1] 区间，与训练时的预处理保持一致
        cropped_list.append(((cropped / 255.0) - 0.5) * 2)
    cropped_arr = np.stack(cropped_list, axis=0)

    # 当前模型不支持批处理，只保留第一张裁剪图。
    # 第 2 部分会演示如何启用批大小 > 0 的情况
    return cropped_arr[None, 0]


# 文本识别后处理：把模型的字符概率张量解码成字符串（CTC 解码）
def recognition_postprocessing(scores: np.ndarray) -> str:
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    # 调整维度顺序，便于按时间步遍历
    scores = np.transpose(scores, (1, 0, 2))

    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += "-"
    # CTC 解码规则：去掉相邻重复字符以及表示空白/背景的 "-"，
    # 得到最终文本输出
    char_list = []
    for i, char in enumerate(text):
        if char != "-" and (not (i > 0 and char == text[i - 1])):
            char_list.append(char)
    return "".join(char_list)


if __name__ == "__main__":
    # 建立与 Triton Inference Server 的连接（HTTP 协议，默认端口 8000）
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # 读取图片并做前处理，构造 Triton 输入对象
    raw_image = cv2.imread("./img1.jpg")
    preprocessed_image = detection_preprocessing(raw_image)

    # InferInput 的第一个参数必须与模型 config 中定义的输入名一致
    detection_input = httpclient.InferInput(
        "input_images:0", preprocessed_image.shape, datatype="FP32"
    )
    # 把 numpy 数据填充进输入对象（binary_data=True 表示以二进制方式传输）
    detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)

    # 向 Triton 发送推理请求：指定模型名与输入
    detection_response = client.infer(
        model_name="text_detection", inputs=[detection_input]
    )

    # 按输出张量名取出检测模型的分数与几何输出，做后处理得到裁剪图片
    scores = detection_response.as_numpy("feature_fusion/Conv_7/Sigmoid:0")
    geometry = detection_response.as_numpy("feature_fusion/concat_3:0")
    cropped_images = detection_postprocessing(scores, geometry, preprocessed_image)

    # 为识别模型构造输入对象，再发起第二次推理请求
    recognition_input = httpclient.InferInput(
        "input.1", cropped_images.shape, datatype="FP32"
    )
    recognition_input.set_data_from_numpy(cropped_images, binary_data=True)

    # 查询服务器
    recognition_response = client.infer(
        model_name="text_recognition", inputs=[recognition_input]
    )

    # 处理识别模型的响应（"308" 是识别模型的输出张量名），解码出文本
    final_text = recognition_postprocessing(recognition_response.as_numpy("308"))

    print(final_text)
