"""
学習されたモデルの指し手正解率などを測定する
"""

import cshogi
from datasets import DatasetDict, load_from_disk
import argparse
import yaml
from .predict_engine import ShogiLLMEngine

def evaluate(engine: ShogiLLMEngine, val_data, count = None) -> tuple[int, int, int, int, int]:
    legal_count = 0
    match_count = 0
    winner_count = 0
    format_error = 0
    total = 0
    for i, (sfen, move_usi, color_str, game_result_str) in enumerate(zip(val_data["sfen"], val_data["move_usi"], val_data["color"], val_data["game_result"])):
        board = cshogi.Board(sfen)
        legal_moves = set(cshogi.move_to_usi(m) for m in board.legal_moves)
        pred = engine.predict(sfen)
        total += 1
        if pred is None:
            print(sfen, "FORMAT ERROR")
            format_error += 1
            continue

        pred_move, _, pred_logit_diff = pred
        if pred_move in legal_moves:
            legal_count += 1
            if pred_move == move_usi:
                match_count += 1
        color = int(color_str)
        game_result = int(game_result_str)
        if pred_logit_diff > 0.0:
            if (color == 0 and game_result == 1) or (color == 1 and game_result == -1):
                winner_count += 1
        else:
            if (color == 0 and game_result == -1) or (color == 1 and game_result == 1):
                winner_count += 1
        
        print(sfen, pred_move, pred_logit_diff, color, game_result)
        if count is not None and i >= count - 1:
            break
    return total, legal_count, match_count, winner_count, format_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config yaml file")
    parser.add_argument("checkpoint", help="model checkpoint directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset: DatasetDict = load_from_disk(config["dataset"])
    engine = ShogiLLMEngine(args.checkpoint, config["formatter"], args.device)
    total, legal_count, match_count, winner_count, format_error = evaluate(engine, dataset[args.dataset_split], args.limit)

    print(f"Total {total} sfens, {legal_count / total * 100:.2f}% legal, {match_count / total * 100:.2f}% bestmove, {winner_count / total * 100:.2f}% winner accuracy, {format_error / total * 100:.2f}% format error")

if __name__ == "__main__":
    main()
