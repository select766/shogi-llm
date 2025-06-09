# shogi-llm
ChessLLMの将棋バージョンを目指す

# 環境構築

uvを使う

```
uv sync
```

# 学習

## データのダウンロードと整形

http://wdoor.c.u-tokyo.ac.jp/shogi/x/ から wdoor2024.7z をダウンロードして `data/floodgate` に配置

```
7z x wdoor2024.7z
```

→ `data/floodgate/2024/*.csa` が生成される

棋譜ファイルを局面と指し手のセットに変換（対局者のレートと終局理由でフィルタ付き）

```
python -m shogi_llm.floodgate_extract_v1 data/floodgate/2024 data/floodgate/extracted_2024.csv
```

2020-2024全部やる場合

```
for i in 202{0,1,2,3,4}; do
pushd data/floodgate
7z x wdoor$i.7z
popd
python -m shogi_llm.floodgate_extract_v1 data/floodgate/$i data/floodgate/extracted_$i.csv
done
```

## データセット生成

10k版はプログラムの動作検証用

```
uv run python -m shogi_llm.generate_dataset_v1 data/floodgate/ds2024_10k data/floodgate/extracted_2024.csv --limit 10000
uv run python -m shogi_llm.generate_dataset_v1 data/floodgate/ds2024 data/floodgate/extracted_2024.csv
uv run python -m shogi_llm.generate_dataset_v1 data/floodgate/ds2020to2024 data/floodgate/extracted_202{0,1,2,3,4}.csv
```

## 学習

yamlで設定ファイルを記述する

```
PYTHONUNBUFFERED=1 uv run python -m shogi_llm.train_v1 config/qwen3-0.6b-t64-winner.yaml | tee qwen3-0.6b-t64-winner.log
```

## 評価

validationデータで指し手の正解率などを計測する

```
uv run python -m shogi_llm.evaluate_model_v1 config/qwen3-0.6b-t64-winner.yaml data/train/qwen3-0.6b-t64-winner/checkpoint-500000 --device cpu --limit 10
```
