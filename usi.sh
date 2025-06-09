#!/bin/bash

cd `dirname $0`
uv run python -m shogi_llm.usi_engine_v1 usi_config/v1.yaml
