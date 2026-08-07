# ============================================================================
# Kafka 消费者封装：轮询订阅主题的消息，把每条消息通过 Ray Serve 句柄
# 提交给 Triton 推理，并把推理结果放入共享队列（供生产者取走）。
# ============================================================================

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from typing import List

from confluent_kafka import KafkaError, KafkaException
from gcn_kafka import Consumer
from ray.serve.handle import DeploymentHandle


# Kafka 消费者：负责订阅主题、轮询消息并触发 Triton 推理
class KafkaConsumer:
    def __init__(
        self,
        config: dict,
        topics: List[str],
        triton_server_handle: DeploymentHandle,
        output_queue: deque,
    ):
        self.config = config
        self.topics = topics
        self.triton_handle = triton_server_handle
        self.output_queue = output_queue

    # 启动消费循环：创建消费者并订阅主题
    def read(self):
        consumer = Consumer(self.config)
        consumer.subscribe(self.topics)
        self._consume_data(consumer)

    # 推理完成后的回调：把结果追加到输出队列，交给生产者发送
    def _infer(self, future):
        print("The custom callback was called.")
        result = future.result()
        self.output_queue.append(result.result())
        print(f"Got: {future.result()}")

    # 核心消费循环：持续轮询消息并提交推理任务
    def _consume_data(self, consumer):
        while True:
            try:
                # 非阻塞轮询（超时 0.1 秒），无消息则继续
                msg = consumer.poll(0.1)
                if not msg:
                    continue
                if msg.error():
                    print(msg.error())
                    # 读到分区末尾属于正常情况，仅打印提示
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print(
                            f"End of partition has been reached {msg.topic()}/{msg.partition()}"
                        )
                    else:
                        raise KafkaException(msg.error())
                print(f"Key: {msg.key()}, Value: {msg.value()}")
                # 用线程池并发提交推理任务（线程数由 KAFKA_CONSUMER_MAX_WORKER_THREADS 控制）
                with ThreadPoolExecutor(
                    max_workers=int(
                        os.environ.get("KAFKA_CONSUMER_MAX_WORKER_THREADS", 1)
                    )
                ) as executor:
                    # 通过 Ray Serve 句柄远程调用 Triton 推理，返回 future
                    future = executor.submit(
                        self.triton_handle.infer.remote, [msg.value()]
                    )
                    # 注册完成回调：把推理结果写入输出队列
                    future.add_done_callback(self._infer)
            except KeyboardInterrupt as e:
                print(f"Keyboard Interrupt Received: {e}")
                break
            except Exception as e:
                print(f"Exception {e}")
        consumer.close()
