import argparse
from typing import Dict
import yaml

import cshogi
from .predict_engine import ShogiLLMEngine

class USIEngine:
    board: cshogi.Board
    engine: ShogiLLMEngine
    options: Dict[str, str]

    def __init__(self, engine: ShogiLLMEngine):
        self.options = {}
        self.engine = engine

    def respond_usi(self, tokens):
        print(f"id name ShogiLLM")
        print("id author select766")
        self.respond_option()
        print("usiok", flush=True)

    def respond_option(self):
        # print("option name BookFile type string default public.bin")
        print("option name OptionFile type string default <empty>")

    def respond_isready(self, tokens):
        print("readyok", flush=True)

    def respond_position(self, tokens):
        # position startpos moves 7g7f 8c8d 7i6h
        if tokens[1] == "startpos":
            self.board = cshogi.Board()
            tokens = tokens[3:]
        elif tokens[1] == "sfen":
            # position sfen ln1g3nl/1ks1g1r2/5sbpp/ppppppp2/7P1/PPPPPSP2/2BG1P2P/1KS4R1/LN1G3NL w - 36 moves 7g7f ...
            self.board = cshogi.Board(" ".join(tokens[2:6]))
            tokens = tokens[7:]
        for move_usi in tokens:
            self.board.push_usi(move_usi)

    def respond_setoption(self, tokens):
        # setoption name BookFile value public.bin
        pass

    def respond_go(self, tokens):
        bestmove = self._search(tokens)
        print(f"bestmove {bestmove}", flush=True)

    def _search(self, tokens):
        legal_moves = list(cshogi.move_to_usi(m) for m in self.board.legal_moves)
        if not legal_moves:
            return "resign"
        # 5手詰めの手があればそれを返し、なければ0
        mate_move = self.board.mate_move(5)
        if mate_move:
            print("info string mate found")
            return cshogi.move_to_usi(mate_move)
        pred = self.engine.predict(self.board.sfen())
        if pred is None:
            print("info string output format is invalid")
            return legal_moves[0]
        pred_move, _, pred_logit_diff = pred
        if pred_move in legal_moves:
            cp = int(pred_logit_diff * 200)
            cp = cp if self.board.turn == 0 else -cp
            print(f"info depth 0 score cp {cp} pv {pred_move}")
            return pred_move
        else:
            print("info string illegal move")
            return legal_moves[0]

    def usi_loop(self):
        while True:
            try:
                msg = input()
            except EOFError:
                break
            tokens = msg.split(" ")
            if tokens[0] == "usi":
                self.respond_usi(tokens)
            elif tokens[0] == "setoption":
                self.respond_setoption(tokens)
            elif tokens[0] == "isready":
                self.respond_isready(tokens)
            elif tokens[0] == "usinewgame":
                pass
            elif tokens[0] == "position":
                self.respond_position(tokens)
            elif tokens[0] == "go":
                self.respond_go(tokens)
            elif tokens[0] == "quit":
                break
            else:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usi_config")
    args = parser.parse_args()

    with open(args.usi_config) as f:
        config = yaml.safe_load(f)
    engine = ShogiLLMEngine(config["llm_checkpoint"], config["formatter"], config["device"])
    usi = USIEngine(engine)
    usi.usi_loop()

if __name__ == "__main__":
    main()
