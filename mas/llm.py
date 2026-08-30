import os

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)

from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .utils import load_config
# Optional local HF backend for Qwen family
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList



# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

# --- Load environment and support DeepSeek/OpenAI compatible providers ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # .env loading is optional; shell may already export envs
    pass
def _resolve_api_env(model_name: str | None):
    """
    Resolve API base URL and key with support for DeepSeek and OpenAI-compatible providers.
    Priority:
      1) Explicit OPENAI_* envs
      2) DeepSeek-specific envs (DEEPSEEK_API_KEY), base 'https://api.deepseek.com'
      3) Fallback: None (mock mode enabled in GPTChat)
    """
    # Prefer OPENAI_* if set
    base = os.getenv("OPENAI_API_BASE")
    key = os.getenv("OPENAI_API_KEY")

    # If not set, try DeepSeek defaults when model suggests deepseek
    name = (model_name or '').lower()
    looks_deepseek = ("deepseek" in name)

    if (not base or not key) and looks_deepseek:
        # DeepSeek official OpenAI-compatible endpoint
        base = base or "https://api.deepseek.com"
        # Prefer DEEPSEEK_API_KEY if provided; otherwise keep OPENAI_API_KEY
        key = key or os.getenv("DEEPSEEK_API_KEY")

    # If still missing base/key, return None values
    if not base or not key:
        return None, None
    return base, key

# Lazily resolve URL/KEY based on model name when GPTChat constructed


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self._mock: bool = False
        self._model_for_provider: str = self._normalize_model_name(model_name)
        # Resolve provider base/key dynamically
        base_url, api_key = _resolve_api_env(model_name)
        # Mock mode if model name starts with 'mock' or missing credentials/client
        if (model_name or '').lower().startswith('mock') or (base_url is None or api_key is None or OpenAI is None):
            self._mock = True
            self.client = None
        else:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )

    @staticmethod
    def _normalize_model_name(model_name: str | None) -> str:
        name = (model_name or '').lower().strip()
        # Map DeepSeek synonyms to official model ids
        if 'deepseek' in name:
            # Reasoner models (aka R1) if explicitly asked
            if 'reasoner' in name or 'r1' in name:
                return 'deepseek-reasoner'
            # Default chat model
            return 'deepseek-chat'
        # OpenAI/others: pass through
        return model_name or ''

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens

        # Mock path: generate deterministic, schema-conformant outputs for offline runs
        if self._mock:
            system_text = next((m.content for m in messages if m.role == 'system'), '')
            user_text = next((m.content for m in messages if m.role == 'user'), '')
            sys_lower = (system_text or '').lower()
            # Simple heuristic: if targeting CAIL2018 schema, return minimal valid JSON
            if 'cail2018' in sys_lower or 'relevant_articles' in system_text or 'term_of_imprisonment' in system_text:
                json_answer = (
                    '{"relevant_articles": [], '
                    '"accusation": [], '
                    '"term_of_imprisonment": {"death_penalty": false, "life_imprisonment": false, "imprisonment": int}}'
                )
                return f"Finish[{json_answer}]"
            # Default mock: echo a brief finish
            return "Finish[OK]"

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5
        wait_time = 1

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_for_provider,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs
                )

                answer = response.choices[0].message.content
                # usage may be None on some providers
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens += getattr(response.usage, 'prompt_tokens', 0) or 0
                    completion_tokens += getattr(response.usage, 'completion_tokens', 0) or 0

                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break

        return ""


class QwenLocalChat(LLM):
    """
    Lightweight local inference wrapper for Qwen/Qwen2.5 models using Transformers.

    Expects `model_name` to be a local path or HF repo id with chat template support,
    or at least compatible with the Qwen <|im_start|> / <|im_end|> format.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Use bfloat16/float16 if available, otherwise fallback to default
        torch_dtype= torch.bfloat16 if torch and torch.cuda.is_available() else None
        # Allow forcing the LLM onto a single device via env (e.g., 'cuda:0' or 'cpu')
        llm_device = os.environ.get("LLM_DEVICE")
        if llm_device and torch:
            # Load without auto-sharding, then move whole model to target device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=None
            )
            try:
                self.model.to(llm_device)
            except Exception:
                # Fallback: keep model as loaded
                pass
        else:
            # Default: use HF auto device_map for multi-GPU sharding
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if torch else None
            )

    def _to_qwen_prompt(self, messages: List[Message]) -> str:
        # Prefer tokenizer's chat template if available
        try:
            # build HF-style messages if template exists
            if hasattr(self.tokenizer, "apply_chat_template"):
                hf_msgs = [{"role": m.role, "content": m.content} for m in messages]
                prompt_str = self.tokenizer.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
                return prompt_str
        except Exception:
            pass
        # Fallback to Qwen 2.5 chat format using special tokens
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m.role}\n{m.content}\n<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _infer_target_device(self):
        """Infer a single device to place small input tensors for generate().
        Prefer the embedding layer device if sharded; otherwise use the first parameter device.
        Always return a torch.device, converting ints/strings appropriately.
        """
        try:
            device_map = getattr(self.model, "hf_device_map", None) or getattr(self.model, "device_map", None)
            if isinstance(device_map, dict) and device_map:
                dev = None
                for key in ("model.embed_tokens", "transformer.wte", "model.decoder.embed_tokens"):
                    if key in device_map:
                        dev = device_map[key]
                        break
                if dev is None:
                    dev = next(iter(device_map.values()))
                # Normalize different forms to torch.device
                if isinstance(dev, int):
                    if torch.cuda.is_available():
                        return torch.device(f"cuda:{dev}")
                    else:
                        return torch.device("cpu")
                if isinstance(dev, str):
                    # e.g., 'cuda:0' or 'cpu'
                    return torch.device(dev)
                if isinstance(dev, torch.device):
                    return dev
            # Fallback: non-sharded model
            try:
                return next(self.model.parameters()).device
            except Exception:
                pass
        except Exception:
            pass
        # Last resort
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        prompt_str = self._to_qwen_prompt(messages)
        inputs = self.tokenizer(prompt_str, return_tensors="pt")
        # 将输入张量迁移到模型的设备（或首个分片/嵌入层设备），避免 CPU/CUDA 警告
        try:
            target_device = self._infer_target_device()
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        except Exception:
            pass

        # Generation configuration
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": max(0.0, temperature or 0.0),
            "do_sample": (temperature or 0.0) > 0.0,
        }
        # Use tokenizer's generic EOS/pad tokens rather than special chat end tag
        # 不显式设置 eos_token_id，避免在含有聊天模板时过早停止；仅设置 pad_token_id
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        # Runtime stopping criteria for early stop on custom strings
        stop_criteria = None
        if stop_strs:
            class StopOnString(StoppingCriteria):
                def __init__(self, tokenizer, stops: List[str]):
                    super().__init__()
                    self.tokenizer = tokenizer
                    self.stops = [s for s in (stops or []) if s]
                    self.prev_len = 0
                def __call__(self, input_ids, scores, **kwargs):
                    try:
                        ids = input_ids[0]
                        new_ids = ids[self.prev_len:]
                        self.prev_len = ids.shape[-1]
                        segment = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                        return any(s in segment for s in self.stops)
                    except Exception:
                        return False
            stop_criteria = StoppingCriteriaList([StopOnString(self.tokenizer, stop_strs)])

        # Generate with optional stopping criteria
        output_ids = self.model.generate(**inputs, **gen_kwargs, stopping_criteria=stop_criteria)
        # 仅解码新生成的 tokens，避免把提示中的 <|im_start|>assistant 之前内容带入
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)

        # Extract assistant segment after the last assistant tag
        # 移除潜在的对话结束标记，不要在首字符即截断
        try:
            # 去掉任何末尾的 <|im_end|> 标记及其后的内容
            if "<|im_end|>" in text:
                text = text.split("<|im_end|>")[0]
        except Exception:
            pass

        # Basic stop trimming
        if stop_strs:
            for s in stop_strs:
                if s and s in text:
                    text = text.split(s)[0]
        return text.strip()


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000
