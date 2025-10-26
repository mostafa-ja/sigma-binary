# Convenience script with example commands used to run some attacks and base performance evaluation in this repository

### Base Performance ###
python -m attacks.run_attacks.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --batch-size 1000
python -m attacks.run_attacks.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --batch-size 1000


# BGA
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose


# PGD-L∞
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --norm linf --step_length 0.001 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --norm linf --step_length 0.01 --max-iterations 1000 --binary-search-steps 4 --verbose


# mimicry
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --verbose


# sigmaBinary_attack
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.4 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --learning-rate 0.3 --threshold 0.3 --binary-search-steps 4 --verbose
