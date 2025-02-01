# This script provides attacks.run_attacks of running different adversarial attacks 
# on various models to evaluate their robustness. The attacks include:
# - Mimicry attack
# - Sigma_zero attack
# - PGD attack
# - CW attack
# - SigmaBinary attack
# The script also includes baseline performance evaluation for reference.

### Base Performance ###
python -m attacks.run_attacks.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --batch-size 1000
python -m attacks.run_attacks.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --batch-size 1000

### mimicry attack ###
python -m attacks.run_attacks.mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --oblivion --batch-size 1000 
python -m attacks.run_attacks.mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --batch-size 1000 --mode deferred --verbose

### sigma_zero attack ###
python -m attacks.run_attacks.sigma_zero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --max-iterations 500 --learning-rate 0.8 --threshold 0.5 --verbose
python -m attacks.run_attacks.sigma_zero_attack --cuda  --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --max-iterations 200 --batch-size 5000 --learning-rate 0.3 --threshold 0.3 --binary-search-steps 4 --initial-const 1.0 --rounded --verbose

### PGD attack ###
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --max-iterations 100 --step_length 0.5 --norm "l2" --rounded --rounding-function "prioritized_binary_rounding"  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --max-iterations 100 --step_length 0.5 --norm "linf" --rounded --rounding-function "thresholded_binary_rounding" --rounding-args '{\"threshold\": 0.5}' --verbose

### CW attack ###
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --max-iterations 100 --learning-rate 0.1 --initial_const_cw 100   --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --max-iterations 100 --learning-rate 0.1 --initial_const_cw 100 --rounded --rounding-function "probabilistic_binary_rounding" --rounding-args '{\"seed\": 42}' --verbose

### sigmaBinary attack ###
python -m attacks.run_attacks.sigmaBinary_attack --cuda  --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 5000 --max-iterations 200 --learning-rate 0.3 --threshold 0.3  --binary-search-steps 4 --initial-const 0.1  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda  --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --max-iterations 400 --learning-rate 0.8 --threshold 0.3 --confidence-bound-primary 1. --confidence-bound-secondary 1.  --confidence-update-interval 20  --verbose













