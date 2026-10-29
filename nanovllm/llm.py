# 对外的入口类：和 vLLM 一样用 LLM(...) 和 llm.generate(...)，实现全在 engine/llm_engine.py 里
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
