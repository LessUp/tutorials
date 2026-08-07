# ============================================================================
# Kafka I/O 示例的主部署文件：
# 用 Ray Serve 把「HTTP 入口（APIIngress）」与「Triton 推理端点
# （TritonKafkaEndpoint）」组织成两个 deployment，并在后台线程中运行
# Kafka 消费者与生产者，构成「Kafka 消息 → Triton 推理 → Kafka 回写」流水线。
# ============================================================================

import json
import os
import threading
from collections import deque
from pprint import pprint
from typing import List

import certifi
import numpy as np
import tritonserver
from fastapi import FastAPI, Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from tritonserver._c.triton_bindings import TRITONSERVER_DataType
from utils.kafka_consumer import KafkaConsumer
from utils.kafka_producer import KafkaProducer, NumpyEncoder

app = FastAPI()


# 若 Kafka 配置启用了 SSL，则自动补上 CA 证书路径（使用 certifi 提供的系统 CA）
def check_ssl_requirement(config: dict) -> dict:
    print(type(config))
    if "SSL" in config.get("security.protocol"):
        if "ssl.ca.location" not in config:
            config["ssl.ca.location"] = certifi.where()
    return config


# 从环境变量读取 Kafka 消费者配置（支持 SASL/OAuthbearer 认证）
def get_consumer_configs() -> dict:
    configs = dict()
    configs["bootstrap.servers"] = os.environ.get("CONSUMER_KAFKA_SERVER", None)
    configs["sasl.mechanisms"] = os.environ.get("CONSUMER_SASL_MECHANISM", None)
    configs["sasl.oauthbearer.method"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_METHOD", None
    )
    configs["sasl.oauthbearer.scope"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_SCOPE", None
    )
    configs["sasl.oauthbearer.client.id"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_CLIENT_ID", None
    )
    configs["sasl.oauthbearer.client.secret"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_CLIENT_SECRET", None
    )
    configs["sasl.oauthbearer.token.endpoint.url"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_TOKEN_ENDPOINT", None
    )
    configs["security.protocol"] = os.environ.get("CONSUMER_SECURITY_PROTOCOL", None)
    return configs


# 从环境变量读取 Kafka 生产者配置（与消费者配置结构对应）
def get_producer_configs() -> dict:
    configs = dict()
    configs["bootstrap.servers"] = os.environ.get("PRODUCER_KAFKA_SERVER", None)
    configs["sasl.mechanisms"] = os.environ.get("PRODUCER_SASL_MECHANISM", None)
    configs["sasl.oauthbearer.method"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_METHOD", None
    )
    configs["sasl.oauthbearer.scope"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_SCOPE", None
    )
    configs["sasl.oauthbearer.client.id"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_CLIENT_ID", None
    )
    configs["sasl.oauthbearer.client.secret"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_CLIENT_SECRET", None
    )
    configs["sasl.oauthbearer.token.endpoint.url"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_TOKEN_ENDPOINT", None
    )
    configs["security.protocol"] = os.environ.get("PRODUCER_SECURITY_PROTOCOL", None)
    return configs


# 打印带分隔线的小标题，便于在日志中定位关键节点
def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


# 网关 deployment：通过 FastAPI 暴露 /infer 与 /health 端点，
# 并在启动时创建 Kafka 消费者与生产者线程，串联整条流水线
@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, distilbert_model_handle: DeploymentHandle) -> None:
        self.handle = distilbert_model_handle
        # 用共享队列连接消费者与生产者：消费者把推理结果放入队列，生产者取出发送到 Kafka
        producer_queue = deque()
        consumer_config = json.loads(os.environ.get("CONSUMER_CONFIGS"))
        consumer_config = check_ssl_requirement(consumer_config)
        # 消费者：订阅 CONSUMER_TOPICS 指定的主题，把每条消息交给 Triton 推理
        consumer = KafkaConsumer(
            consumer_config,
            (os.environ.get("CONSUMER_TOPICS", "")).split(","),
            self.handle,
            producer_queue,
        )
        producer_config = json.loads(os.environ.get("PRODUCER_CONFIGS"))
        producer_config = check_ssl_requirement(producer_config)
        # 生产者：把队列中的推理结果写入 PRODUCER_TOPIC 主题
        producer = KafkaProducer(
            producer_config, os.environ.get("PRODUCER_TOPIC", ""), producer_queue
        )
        print("Starting Producer")
        # 生产者线程：持续从队列取结果并发送到 Kafka（示例用单线程，生产可换线程池）
        threading.Thread(
            target=producer.send_data, daemon=False
        ).start()  # convert to thread pool
        print("Starting Consumer")
        # 消费者线程：持续轮询 Kafka 消息并触发推理
        threading.Thread(target=consumer.read, daemon=False).start()

    # HTTP POST /infer：把请求体中的 sentences 交给 Triton 推理端点
    @app.post("/infer")
    async def classify(self, request: Request):  # sentence: str):
        data = await request.json()
        print(data)
        # 通过 Ray Serve 句柄远程调用 Triton 端点的 infer 方法
        return await self.handle.infer.remote(data.get("sentences"))

    # HTTP GET /health：健康检查端点
    @app.get("/health")
    async def health(self):
        return "OK"


# Triton 推理端点 deployment：每个副本内嵌一个进程内 Triton server，
# 负责加载模型并执行推理（申请 1 块 GPU，可自动扩缩到 2 个副本）
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class TritonKafkaEndpoint:
    def __init__(self):
        # 从环境变量读取模型名、输入名与模型仓库路径
        self.model_name = os.environ.get("MODEL_NAME")
        self.model_input_name = os.environ.get("MODEL_INPUT_NAME")
        self.model_repository = os.environ.get("MODEL_REPOSITORY")
        # 创建进程内 Triton server：显式模型控制模式 + 开启日志与各项指标
        self._triton_server = tritonserver.Server(
            model_repository=self.model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=True,
            metrics=True,
            gpu_metrics=True,
            cpu_metrics=True,
        )
        # 阻塞等待 server 完全就绪后再继续
        self._triton_server.start(wait_until_ready=True)
        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        # 显式控制模式下模型需手动 load，这里检查并加载模型
        if not self._triton_server.model(self.model_name).ready():
            try:
                self._tokenizer_model = self._triton_server.load(self.model_name)

                if not self._tokenizer_model.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't load tokenizer model!")
                print(
                    f"Please ensure dependencies are met and you have set the environment variables if any {error}"
                )

    # 推理方法：把消息列表作为输入送给 Triton，把响应序列化为 JSON 返回
    def infer(self, message: List[str]):
        responses = self._triton_server.model(self.model_name).infer(
            inputs={self.model_input_name: np.array(message)}
        )
        result = list()
        # 遍历推理响应，按数据类型把输出转换为可 JSON 序列化的格式
        for response in responses:
            out = dict()
            for output, value in response.outputs.items():
                # 字节类型（字符串）输出转为字符串数组，其余（数值）张量经 DLPack 转 numpy
                if value.data_type == TRITONSERVER_DataType.BYTES:
                    out[output] = value.to_string_array()
                else:
                    out[output] = np.from_dlpack(value)
            json_message = response.__dict__
            json_message["outputs"] = out
            json_message["model"] = json_message["model"].__dict__
            json_message["model"].pop("_server", None)
            result.append(json.dumps(json_message, cls=NumpyEncoder))
        return {"result": result}


# 构建两个 deployment 之间的依赖：HTTP 入口持有 Triton 端点的句柄
triton_handle = TritonKafkaEndpoint.bind()
entrypoint = APIIngress.bind(triton_handle)
