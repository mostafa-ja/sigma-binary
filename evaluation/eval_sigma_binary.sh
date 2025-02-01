# This script compares various attack methods to demonstrate the effectiveness of our proposed Sigma-Binary Attack  
# and reproduce the results of the paper (RQ2). The script executes the following commands:

python -m attacks.run_attacks.mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --batch-size 1000 --verbose 
python -m attacks.run_attacks.mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM  --batch-size 1000 --verbose 
python -m attacks.run_attacks.mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --batch-size 1000 --verbose 
python -m attacks.run_attacks.mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --batch-size 1000 --verbose 

python -m attacks.run_attacks.sigma_zero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --max-iterations 10000 --learning-rate 0.5 --threshold 0.1 --rounded --verbose
python -m attacks.run_attacks.sigma_zero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM  --max-iterations 10000 --learning-rate 0.8 --threshold 0.5 --rounded --verbose
python -m attacks.run_attacks.sigma_zero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --max-iterations 10000 --learning-rate 3.8 --threshold 0.5 --rounded --verbose
python -m attacks.run_attacks.sigma_zero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --max-iterations 10000 --learning-rate 0.3 --threshold 0.3 --rounded --verbose

python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --max-iterations 10000 --learning-rate 0.001 --initial_const_cw 100 --rounded --rounding-function "prioritized_binary_rounding"  --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM  --max-iterations 10000 --learning-rate 0.05 --initial_const_cw 0.1 --rounded --rounding-function "prioritized_binary_rounding"  --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --max-iterations 10000 --learning-rate 0.05 --initial_const_cw 0.1 --rounded --rounding-function "prioritized_binary_rounding"  --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --max-iterations 10000 --learning-rate 0.02 --initial_const_cw 10 --initial_const_penalty 0.1 --rounded --rounding-function "prioritized_binary_rounding"  --verbose

python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --max-iterations 10000 --step_length 1.0 --norm "l1"  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM  --max-iterations 10000 --step_length 1.0 --norm "l1"  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --max-iterations 10000 --step_length 1.0 --norm "l1"  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --max-iterations 10000 --step_length 1.0 --norm "l1"  --verbose


python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --max-iterations 10000 --step_length 0.001 --norm "l2" --rounded  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM  --max-iterations 10000 --step_length 0.02 --norm "l2" --rounded  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --max-iterations 10000 --step_length 1.0 --norm "l2" --rounded  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --max-iterations 10000 --step_length 0.005 --norm "l2" --rounded  --verbose


python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  --max-iterations 10000 --step_length 0.001 --norm "linf" --rounded  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM  --max-iterations 10000 --step_length 0.001 --norm "linf" --rounded  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA  --max-iterations 10000 --step_length 0.01 --norm "linf" --rounded  --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --max-iterations 10000 --step_length 0.001 --norm "linf" --rounded  --verbose


python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --confidence-bound-primary 0.5 --confidence-update-interval 50 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --max-iterations 10000 --learning-rate 0.85 --threshold 0.6 --confidence-bound-primary 1.7 --confidence-update-interval 20 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --max-iterations 10000 --learning-rate 4.2 --threshold 0.4 --confidence-bound-primary 0.8 --confidence-update-interval 50 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --max-iterations 10000 --learning-rate 0.3 --threshold 0.3 --confidence-bound-primary 1.4 --confidence-bound-secondary 1.4  --confidence-update-interval 120   --verbose


