#!/bin/bash
set -e

mkdir -p N_nodes6_N_layers1_out && python3 ../../run_performance.py N_nodes6_N_layers1/N_nodes6_N_layers1.yaml > N_nodes6_N_layers1_out/output.log
mkdir -p N_nodes6_N_layers2_out && python3 ../../run_performance.py N_nodes6_N_layers2/N_nodes6_N_layers2.yaml > N_nodes6_N_layers2_out/output.log
mkdir -p N_nodes9_N_layers1_out && python3 ../../run_performance.py N_nodes9_N_layers1/N_nodes9_N_layers1.yaml > N_nodes9_N_layers1_out/output.log
mkdir -p N_nodes9_N_layers2_out && python3 ../../run_performance.py N_nodes9_N_layers2/N_nodes9_N_layers2.yaml > N_nodes9_N_layers2_out/output.log
mkdir -p N_nodes12_N_layers1_out && python3 ../../run_performance.py N_nodes12_N_layers1/N_nodes12_N_layers1.yaml > N_nodes12_N_layers1_out/output.log
mkdir -p N_nodes12_N_layers2_out && python3 ../../run_performance.py N_nodes12_N_layers2/N_nodes12_N_layers2.yaml > N_nodes12_N_layers2_out/output.log
