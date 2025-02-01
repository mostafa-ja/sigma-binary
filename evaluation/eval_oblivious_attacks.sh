# This script evaluates various defenses against the sigma_binary attack in oblivious attack mode.  
# to demonstrate the robustness of strong defenses against adversarial attacks  
# and reproduce the results of the paper (RQ4). The script executes the following commands:

python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --oblivion --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --oblivion --mode deferred --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --oblivion --max-iterations 10000 --learning-rate 0.5 --threshold 0.2  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --oblivion --mode deferred --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --oblivion --batch-size 1000 --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --oblivion --mode deferred --batch-size 1000 --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --oblivion --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --oblivion --mode deferred --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --oblivion --max-iterations 10000 --learning-rate 0.5 --threshold 0.2  --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --oblivion --mode deferred --max-iterations 10000 --learning-rate 0.5 --threshold 0.2 --verbose

