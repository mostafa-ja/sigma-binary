# This script runs baseline performance evaluations for various defense defenses to reproduce the results of the paper (RQ2).
# The script runs the following commands:
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN  
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE  --mode deferred
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA  --mode deferred
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus  --mode deferred
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN  --mode deferred
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD 
python -m defenses.models_base_performance --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD  --mode deferred
