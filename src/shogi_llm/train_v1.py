from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk
from trl import DataCollatorForCompletionOnlyLM
import torch
from trl import SFTTrainer, SFTConfig
import argparse
import yaml


def get_formatter(n_think=0, append_winner=False):
    def formatting_prompts_func(example):
        output_texts = []
        # " think" を挿入することで考える時間を与える
        think_tokens = " think" * n_think
        for i in range(len(example["sfen"])):
            # ここで最後のトークンを<|endoftext|>にすると、loss計算で<|endoftext|>の部分が計算されない＝テキスト終了のタイミングがわからなくなる
            if append_winner:
                # 勝ったプレイヤーの記号を推定する（手番に関わらず盤面の向きは固定のため、手番側が勝つかどうかよりプレイヤーの色のほうが推定しやすいと期待）
                color = int(example["color"][i])
                game_result = int(example["game_result"][i])
                assert color in [0, 1]
                # TORYO/KACHIだけを抽出しているので引き分けは入っていない
                assert game_result in [-1, 1]
                if color == 0:
                    win_info = " B" if game_result == 1 else " W"
                else:
                    win_info = " W" if game_result == 1 else " B"
            else:
                win_info = ""
            text = f"{example['sfen'][i]}{think_tokens}<|im_start|>{example['move_usi'][i]}{win_info}<|im_end|>"
            output_texts.append(text)
        return output_texts
    return formatting_prompts_func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config yaml file")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset: DatasetDict = load_from_disk(config["dataset"])
    model_name = config["base_model"]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorForCompletionOnlyLM(response_template="<|im_start|>", tokenizer=tokenizer)
    args = SFTConfig(
        **config["sft_config"]
    )

    trainer = SFTTrainer(
        model,
        args=args,
        train_dataset=dataset["train"],
        formatting_func=get_formatter(**config["formatter"]),
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
