from transformers import AutoTokenizer

class TokenizerGlue:
    def __init__(self, model_id: str = "unsloth/Llama-3.2-1B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        if "gemma-4" in model_id.lower() or "gemma_4" in model_id.lower() or "gemma4" in model_id.lower():
            self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<turn|>")
        elif "gemma" in model_id.lower():
            self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        else:
            self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
        
    def decode(self, token_ids: list[int] | int) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids)
        
    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> list[int]:
        res = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt
        )
        if isinstance(res, dict) or hasattr(res, "input_ids"):
            return res["input_ids"]
        return res
