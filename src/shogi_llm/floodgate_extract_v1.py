import argparse
from pathlib import Path
from typing import NamedTuple
from tqdm import tqdm
import cshogi
import cshogi.CSA

class SingleRecord(NamedTuple):
    sfen: str
    move_usi: str
    color: int # 手番(black=0, white=1)
    score: int # 手番から見た評価値
    game_result: int # 手番側が勝ったかどうか(勝ち:1, 負け: -1, 引き分け: 0)

def parse_one_file(path: Path, min_rate: float, endgames: list[str]) -> list[SingleRecord]:
    csa = cshogi.CSA.Parser()
    csa.parse_csa_file(str(path))

    records = []

    if csa.endgame not in endgames:
        return []
    if not all(rating is not None and rating >= min_rate for rating in csa.ratings):
        return []

    win_color = csa.win - 1 # csa.winは先手勝ち=1, 後手勝ち=2, それ以外=0
    board = cshogi.Board(csa.sfen)
    for move, sente_score in zip(csa.moves, csa.scores):
        move_usi = cshogi.move_to_usi(move)
        score = sente_score if board.turn == 0 else -sente_score # csaでは先手から見た評価値が入っているので手番側に補正
        if win_color < 0:
            game_result = 0
        else:
            game_result = 1 if board.turn == win_color else -1 # 手番側が勝ったかどうかに変換
        records.append(SingleRecord(sfen=board.sfen(), move_usi=move_usi, color=board.turn, score=score, game_result=game_result))
        board.push(move)
    return records

def parse_all_files(dst: Path, kifu_dir: Path, min_rate: float, endgames: list[str]):
    valid_kifu_count = 0
    total_records = 0
    with open(dst, "w") as f:
        for path in tqdm(sorted(kifu_dir.glob("*.csa"))): # 順序をファイルシステム依存にしないためにsort
            try:
                records = parse_one_file(path, min_rate=min_rate, endgames=endgames)
                if len(records) > 0:
                    valid_kifu_count += 1
                for r in records:
                    f.write(f"{r.sfen},{r.move_usi},{r.color},{r.score},{r.game_result}\n")
                    total_records += 1
            except Exception as e:
                print(f"Error: {path}, {e}")
    
    print(f"Total {valid_kifu_count} valid kifus, {total_records} records")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory path to floodgate csa files")
    parser.add_argument("output", help="output file path")
    parser.add_argument("--min_rate", type=int, default=3500)
    parser.add_argument("--endgames", default="%TORYO,%KACHI")
    args = parser.parse_args()
    parse_all_files(Path(args.output), Path(args.input_dir), args.min_rate, args.endgames.split(","))

if __name__ == "__main__":
    main()
