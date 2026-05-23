#!/bin/bash
# OpenAgriNet Data Pipeline — end-to-end demo
# Runs: synthetic data → ingest → PII → SFT export → split → LoRA dry-run
set -e

echo ""
echo "============================================================"
echo "  OpenAgriNet Data Pipeline — End-to-End Demo"
echo "============================================================"
echo ""

echo "▶ Step 1/6: Generate synthetic agricultural agent logs (300 sessions)"
python -m scripts.generate_synthetic_logs
echo ""

echo "▶ Step 2/6: Ingest + segment into trajectories with metadata"
python -m src.ingest --input data/synthetic_logs/raw_logs.jsonl --output data/processed/trajectories.jsonl
echo ""

echo "▶ Step 3/6: Apply 3-layer PII redaction (rules + Presidio + dedup)"
python -m scripts.run_pii
echo ""

echo "▶ Step 4/6: Export SFT JSONL with Qwen2.5 chat template"
python -m src.export_sft
echo ""

echo "▶ Step 5/6: Split train/eval/preference with MinHash near-duplicate detection"
python -m src.splitter
echo ""

echo "▶ Step 6/6: LoRA dry-run — proves data is training-ready"
python scripts/lora_dryrun.py
echo ""

echo "============================================================"
echo "  Pipeline complete. See data/processed/ and data/exports/"
echo "============================================================"