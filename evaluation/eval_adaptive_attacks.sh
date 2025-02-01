# This script evaluates various defenses against the sigma_binary attack in adaptive attack mode  
# to assess the robustness of strong defenses against adversarial attacks  
# and reproduce the results of the paper (RQ5). The script executes the following commands:

python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --confidence-bound-primary 0.5 --confidence-update-interval 50 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --max-iterations 10000 --learning-rate 0.85 --threshold 0.6 --confidence-bound-primary 1.7 --confidence-update-interval 20 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --max-iterations 10000 --learning-rate 4.2 --threshold 0.4 --confidence-bound-primary 0.8 --confidence-update-interval 50 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 500 --max-iterations 10000 --learning-rate 0.5 --threshold 0.2  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --max-iterations 10000 --learning-rate 0.5 --threshold 0.2  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --max-iterations 10000 --learning-rate 0.7 --threshold 0.2  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --max-iterations 10000 --learning-rate 0.6 --threshold 0.2  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --max-iterations 10000 --learning-rate 0.3 --threshold 0.3 --confidence-bound-primary 1.4 --confidence-bound-secondary 1.4  --confidence-update-interval 120   --verbose
