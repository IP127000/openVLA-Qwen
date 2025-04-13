"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
    
class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 271, min_action = -1.0, max_action = 1.0
        # self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # print('initializing action tokenizer, the origin len(tokenizer) is:', len(self.tokenizer))
        self.origin_len = len(self.tokenizer)
        # 创建等间距的区间和计算区间中心
        self.bins = np.linspace(min_action, max_action, self.n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 添加新的动作 tokens 到 tokenizer
        self.action_tokens = [f"<ACTION_{i}>" for i in range(self.n_bins)]
        self.tokenizer.add_tokens(self.action_tokens)
        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(self.action_tokens)
        self.new_len = len(self.tokenizer)

        # print('initializing action tokenizer, the new len(tokenizer) is:', len(self.tokenizer))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins) - 1
        discretized_action = np.clip(discretized_action, a_min=0, a_max=self.n_bins - 1)

        action_token_ids = np.array(self.action_token_ids)[discretized_action]
        # print(action_token_ids)
        if len(action_token_ids.shape) == 1:
            # return self.tokenizer.decode(action_token_ids)
            return action_token_ids
        else:
            # return self.tokenizer.batch_decode(action_token_ids.tolist())
            return [ids.tolist() for ids in action_token_ids]

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        token_id_to_action_index = {tid: idx for idx, tid in enumerate(self.action_token_ids)}
        vectorized_lookup = np.vectorize(token_id_to_action_index.get)
        action_indices = vectorized_lookup(action_token_ids)
        return self.bin_centers[action_indices]

    def actionValue_to_actionName(self,action):
        action_token_ids=self.__call__(action)
        actionName=self.tokenizer.decode(action_token_ids)
        return actionName

    def actionName_to_actionValue(self,actionName):
        action_ids=self.tokenizer(actionName)
        action_ids=action_ids["input_ids"]
        actionValue=self.decode_token_ids_to_actions(action_ids)
        return actionValue

    def str_to_mllm(self, result_str):
        new_result=result_str
        for token in self.action_tokens:
            value=self.actionName_to_actionValue(token)
            new_result=new_result.replace(token, str(value[0]))
        return new_result

    def value_to_mllm(self, result_str):
        new_result=result_str
        data=[]
        for token in self.action_tokens:
            if new_result.startswith(token):
                value=self.actionName_to_actionValue(token)
                data.append(float(value))
            elif token in new_result:
                value=self.actionName_to_actionValue(token)
                data.append(float(value))
        return data

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def num_new_tokens(self) -> int:
        return self.new_len - self.origin_len

    @property
    def action_vocab_size(self) -> int:
        return len(self.action_tokens)    
