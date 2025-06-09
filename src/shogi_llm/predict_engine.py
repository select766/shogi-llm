import re
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ShogiLLMEngine:
    def __init__(self, checkpoint_dir, formatter_config, device):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self._ind_b = self.tokenizer(" B")["input_ids"][0]
        self._ind_w = self.tokenizer(" W")["input_ids"][0]
        self.formatter_config = formatter_config
        self._eos_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index("<|im_end|>")]

    def predict(self, sfen: str) -> Optional[tuple[str, str, float]]:
        input_text = f"{sfen}{' think'*self.formatter_config['n_think']}<|im_start|>"
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            # GenerateDecoderOnlyOutput
            generated = self.model.generate(
                **inputs,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = self._eos_token_id,
                max_new_tokens = 10,
                do_sample = False,#false: temperature=0
                # temperature = 0.8,
                # top_p = 0.9,
                # top_k = 0,
                repetition_penalty = 1.0, # no penalty (penaltyを与えると明らかに精度が下がる)
                num_beams = 1,
                return_dict_in_generate=True,
                output_logits=True,
            )

        if len(generated.sequences[0]) < 3:
            return None
        generated_text = self.tokenizer.decode(generated.sequences[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        m = re.match("^((?:[1-9a-z]{4}\\+?)|(?:[A-Z]\\*[1-9a-z]{2})) ([BW])$", generated_text)
        if m is None:
            # format error
            return None
        winner_pred_token = generated.logits[-2]
        logit_diff = winner_pred_token[0, self._ind_b].item() - winner_pred_token[0, self._ind_w].item()
        # move_usi, winner (B or W), logit_diff of winner (>0: black wins, <0: white wins)
        return m.group(1), m.group(2), logit_diff
