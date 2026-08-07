# ============================================================================
# Kafka 生产者封装：从共享队列中取出推理结果，序列化后写入指定主题。
# ============================================================================

import json
from collections import deque
from datetime import datetime

import numpy as np
from confluent_kafka.serialization import StringSerializer
from gcn_kafka import Producer


# JSON 编码器：把 numpy 数组转换为 Python list，保证结果可 JSON 序列化
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Kafka 生产者：持续从消息队列取数据并发送到目标主题
class KafkaProducer:
    def __init__(self, config: dict, topic: str, message_queue: deque):
        self.config = config
        self.topics = topic
        self.message_queue = message_queue
        self.serializer = StringSerializer("utf_8")

    # 启动生产循环
    def send_data(self):
        producer = Producer(self.config)
        self._produce(producer)

    # 核心生产循环：轮询队列，有消息就 produce + flush
    def _produce(self, producer):
        # 投递结果回调：报告消息发送成功或失败
        def delivery_report(err, msg):
            """
            Reports the failure or success of a message delivery.
            Args:
                 err (KafkaError): The error that occurred on None on success.
                msg (Message): The message that was produced or failed.
            """
            if err is not None:
                print(f"Delivery failed for User record {msg.key()}: {err}")
                return
            print(
                f"User record successfully produced to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

        while True:
            # 非阻塞触发回调事件（如投递结果回调）
            producer.poll(0.0)
            try:
                # 队列非空时取出一个推理结果，以当前时间戳为 key 发送
                if self.message_queue.__len__() > 0:
                    producer.produce(
                        topic=self.topics,
                        key=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                        value=self.serializer(
                            json.dumps(self.message_queue.pop(), cls=NumpyEncoder)
                        ),
                        on_delivery=delivery_report,
                    )
                    # flush 确保消息送达 broker（会阻塞到队列清空）
                    producer.flush()
            except KeyboardInterrupt as e:
                print(f"Keyboard Interrupt received {e}")
                break
            except Exception as e:
                print(f"Error while producing the message {e}")
            finally:
                producer.flush()
        producer.close()
