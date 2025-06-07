"""
HuggingfaceのDatasetに変換
"""

import argparse
from datasets import Dataset, DatasetDict

def get_generator(paths, limit=None):
    def gen():
        count = 0
        for path in paths:
            with open(path) as f:
                for line in f:
                    sfen, move_usi, color, score, game_result = line.rstrip().split(",")
                    yield {"sfen": sfen, "move_usi": move_usi, "color": color, "score": score, "game_result": game_result}
                    count += 1
                    if limit is not None and count >= limit:
                        return
    return gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="データセットの出力パス(ディレクトリが生成される)")
    parser.add_argument("input_csv", nargs="+", help="局面CSVファイル(複数個指定可能)")
    parser.add_argument("--limit", type=int, help="局面数制限")
    args = parser.parse_args()

    gen = get_generator(args.input_csv, args.limit)
    ds: Dataset = Dataset.from_generator(gen)
    ds = ds.shuffle(seed=1)
    # train:validation:test = 0.8:0.1:0.1
    train_tmp = ds.train_test_split(test_size=0.01, seed=1)
    val_test = train_tmp["test"].train_test_split(test_size=0.5, seed=1)
    ds_dict = DatasetDict({"train": train_tmp["train"], "validation": val_test["train"], "test": val_test["test"]})

    ds_dict.save_to_disk(args.output)

if __name__ == "__main__":
    main()
