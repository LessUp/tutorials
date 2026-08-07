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

import json
from collections import defaultdict
from typing import DefaultDict, Dict, List

import torch
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.integrations.trtllm import build_trtlmm_tokenizer_data
from outlines.fsm.guide import RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import adapt_tokenizer
from pydantic import BaseModel
from transformers import AutoTokenizer


# 嵌套 schema：魔杖信息（木头、杖芯、长度）
class WandFormat(BaseModel):
    """Represents the format of a wand description.

    Attributes:
        wood (str): The type of wood used in the wand.
        core (str): The core material of the wand.
        length (float): The length of the wand.
    """

    wood: str
    core: str
    length: float


# 期望的输出 schema：约束解码将强制 LLM 输出符合该结构的 JSON
class AnswerFormat(BaseModel):
    """Represents the output format, which LLM should follow.

    Attributes:
        name (str): The name of the person.
        house (str): The house affiliation of the person (e.g., Gryffindor).
        blood_status (str): The blood status (e.g., pure-blood).
        occupation (str): The occupation of the person.
        alive (str): Whether the person is alive.
        wand (WandFormat): The wand information.
    """

    name: str
    house: str
    blood_status: str
    occupation: str
    alive: str
    wand: WandFormat


# 基于 LM Format Enforcer 的 logits 后处理器：在采样前把不合 schema 的 token 概率屏蔽
class LMFELogitsProcessor:
    """
    The class implementing logits post-processor via LM Format Enforcer.
    """

    # 处理器名字，客户端通过 logits_post_processor_name 请求参数指定
    PROCESSOR_NAME = "lmfe"

    def __init__(self, tokenizer_dir, schema):
        # 初始化需要 tokenizer（用于把 schema 状态映射到 token 集合）与 JSON schema
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        self.eos_token = tokenizer.eos_token_id
        tokenizer_data = build_trtlmm_tokenizer_data(tokenizer)
        # TokenEnforcer provides a token filtering mechanism,
        # given a tokenizer and a CharacterLevelParser.
        # ref: https://github.com/noamgat/lm-format-enforcer/blob/fe6cbf107218839624e3ab39b47115bf7f64dd6e/lmformatenforcer/tokenenforcer.py#L32
        # JsonSchemaParser 把 JSON schema 编译成字符级解析器，TokenEnforcer 据此过滤 token
        self.token_enforcer = TokenEnforcer(tokenizer_data, JsonSchemaParser(schema))

    # 根据当前已生成的 token 序列，返回下一步允许采样的 token 集合
    def get_allowed_tokens(self, ids):
        def _trim(ids):
            return [x for x in ids if x != self.eos_token]

        allowed = self.token_enforcer.get_allowed_tokens(_trim(ids[0]))
        return allowed

    # TensorRT-LLM 在每个解码步回调：req_id 请求 id，logits 当前各 token 分数，ids 已生成序列
    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        ids: List[List[int]],
        stream_ptr: int,
    ):
        # Create a mask with negative infinity to block all tokens initially.
        # 初始全部屏蔽（-inf），再对允许的 token 位置置 0，实现硬约束
        mask = torch.full_like(logits, fill_value=float("-inf"), device=logits.device)
        allowed = self.get_allowed_tokens(ids)
        # Update the mask to zero for allowed tokens,
        # allowing them to be selected.
        mask[:, :, allowed] = 0
        # 在指定的 CUDA 流上把 mask 加到 logits 上，屏蔽非法 token
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits += mask


# 基于 Outlines 的 logits 后处理器：把 JSON schema 编译成正则 + FSM，逐步约束解码
class OutlinesLogitsProcessor:
    """
    The class implementing logits post-processor via Outlines.
    """

    # 处理器名字，客户端通过 logits_post_processor_name 请求参数指定
    PROCESSOR_NAME = "outlines"

    def __init__(self, tokenizer_dir, schema):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        tokenizer = adapt_tokenizer(tokenizer)
        # 从 JSON schema 生成等价的正则，再用 RegexGuide 构建有限状态机（FSM）
        regex_string = build_regex_from_schema(json.dumps(schema))
        self.fsm = RegexGuide(regex_string, tokenizer)
        # 按请求（seq_id）保存 FSM 当前状态；mask_cache 缓存已算过的状态对应的掩码
        self._fsm_state: DefaultDict[int, int] = defaultdict(int)
        self.mask_cache: Dict[int, torch.Tensor] = {}
        # By default, TensorRT-LLM includes request query into the output.
        # Outlines should only look at generated outputs, thus we'll keep
        # track of the request's input prefix.
        # 记录输入前缀长度：FSM 只约束模型新生成的 token，不约束 prompt 本身
        self._prefix = [-1]

    # TensorRT-LLM 每个解码步回调：推进 FSM 状态并返回当前步的合法 token 掩码
    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        ids: List[List[int]],
        stream_ptr: int,
    ):
        seq_id = None
        # If the prefix token IDs have changed we assume that we are dealing
        # with a new sample and reset the FSM state
        # 前缀变化说明是新请求（或与已处理请求相同），重置 FSM 状态
        if (
            ids[0][: len(self._prefix)] != self._prefix
            # handling edge case, when the new request is identical to already
            # processed
            or len(ids[0][len(self._prefix) :]) == 0
        ):
            self._fsm_state = defaultdict(int)
            self._prefix = ids[0]
            seq_id = hash(tuple([]))

        else:
            # Remove the prefix token IDs from the input token IDs,
            # because the FSM should only be applied to the generated tokens
            # 去掉前缀，只对生成部分推进 FSM：根据上一步状态 + 最新 token 得到新状态
            ids = ids[0][len(self._prefix) :]
            last_token = ids[-1]
            last_seq_id = hash(tuple(ids[:-1]))
            seq_id = hash(tuple(ids))
            self._fsm_state[seq_id] = self.fsm.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token
            )

        state_id = self._fsm_state[seq_id]
        # 该状态首次出现时计算合法 token 掩码并缓存，后续直接复用（省去重复计算）
        if state_id not in self.mask_cache:
            allowed_tokens = self.fsm.get_next_instruction(
                state=self._fsm_state[seq_id]
            ).tokens
            # Create a mask with negative infinity to block all
            # tokens initially.
            mask = torch.full_like(
                logits, fill_value=float("-inf"), device=logits.device
            )
            # Update the mask to zero for allowed tokens,
            # allowing them to be selected.
            mask[:, :, allowed_tokens] = 0
            self.mask_cache[state_id] = mask
        else:
            mask = self.mask_cache[state_id]

        # 在指定的 CUDA 流上把掩码加到 logits，非法 token 的概率变为 -inf
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits += mask
