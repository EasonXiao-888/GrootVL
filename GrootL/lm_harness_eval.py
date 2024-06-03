import torch

import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM, MambaForCausalLM, MambaConfig
from tree_scanning import tree_scanning_algorithm

import numpy as np
import random

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

from peft import PeftModel

@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained=None, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.float16):
        LM.__init__(self)
        self.peft = False
        
        transformers.models.mamba.modeling_mamba.MambaMixer.slow_forward =  tree_scanning_algorithm
        self._model = AutoModelForCausalLM.from_pretrained(pretrained).to(device=device).to(dtype=dtype)
        print("loading success")
        if self.peft:
            self._model = PeftModel.from_pretrained(
                self._model,
                pretrained,
                torch_dtype=dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
# 设置随机数种子
    setup_seed(20)
    cli_evaluate()
