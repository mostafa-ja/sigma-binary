# This script compares various attack methods to demonstrate the effectiveness of our proposed Sigma-Binary Attack  
# and reproduce the results of the paper (RQ1). The script executes the following commands:


# BCA
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BCA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps 4 --verbose


# BGA
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.BGA_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps 4 --verbose


# Grosse
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.Grosse_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps 4 --verbose


# rFGSM
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --step-length 0.001 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --step-length 0.01 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --step-length 0.01 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-length 0.005 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-length 0.001 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-length 0.01 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-length 0.05 --verbose
python -m attacks.run_attacks.rFGSM_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps 4 --step-length 0.001 --verbose


# PGD-L1
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --norm l1 --step_length 1.0 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --norm l1 --step_length 1.0 --max-iterations 1000 --binary-search-steps 4 --verbose


# PGD-L2
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --norm l2 --step_length 1.2 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --norm l2 --step_length 0.4 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --norm l2 --step_length 0.4 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --norm l2 --step_length 0.6 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --norm l2 --step_length 2.5 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --norm l2 --step_length 2.5 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --norm l2 --step_length 1.0 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --norm l2 --step_length 1.2 --max-iterations 1000 --binary-search-steps 4 --verbose


# PGD-L∞
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --norm linf --step_length 0.001 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --norm linf --step_length 0.005 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --norm linf --step_length 0.02 --max-iterations 1000 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --norm linf --step_length 0.005 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --norm linf --step_length 0.005 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --norm linf --step_length 0.01 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --norm linf --step_length 0.05 --max-iterations 1000 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.PGD_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --norm linf --step_length 0.01 --max-iterations 1000 --binary-search-steps 4 --verbose


# iMax
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --step-lengths '{"L1":1.0,"L2":1.2,"Linf":0.001}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --step-lengths '{"L1":1.0,"L2":0.4,"Linf":0.005}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --step-lengths '{"L1":1.0,"L2":0.4,"Linf":0.02}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":0.6,"Linf":0.005}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":2.5,"Linf":0.005}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":2.5,"Linf":0.01}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":1.0,"Linf":0.05}' --verbose
python -m attacks.run_attacks.iMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":1.2,"Linf":0.01}' --verbose


# SMA
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --step-lengths '{"L1":1.0,"L2":0.4,"Linf":0.05}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --step-lengths '{"L1":1.0,"L2":1.4,"Linf":0.05}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --step-lengths '{"L1":1.0,"L2":2.6,"Linf":0.05}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":0.6,"Linf":0.01}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":0.4,"Linf":0.01}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":0.8,"Linf":0.005}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":1.2,"Linf":0.05}' --verbose
python -m attacks.run_attacks.StepwiseMax_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps 4 --step-lengths '{"L1":1.0,"L2":0.6,"Linf":0.01}' --verbose


# mimicry
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --verbose
python -m attacks.run_attacks.Mimicry_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --verbose


# CW
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --learning-rate 0.01 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --learning-rate 0.2 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --learning-rate 0.1 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --binary-search-steps-cw 8 --binary-search-steps-penalty 4 --learning-rate 0.01 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --binary-search-steps-cw 8 --binary-search-steps-penalty 4 --learning-rate 0.05 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --binary-search-steps-cw 8 --binary-search-steps-penalty 4 --learning-rate 0.01 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --binary-search-steps-cw 8 --binary-search-steps-penalty 4 --learning-rate 0.1 --verbose
python -m attacks.run_attacks.CW_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --binary-search-steps-cw 8 --binary-search-steps-penalty 4 --learning-rate 0.01 --verbose


# sigmaZero_attack
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --learning-rate 0.85 --threshold 0.6 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --learning-rate 4.0 --threshold 0.3 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.4 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.2 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.4 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --learning-rate 0.3 --threshold 0.3 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaZero_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --learning-rate 0.5 --threshold 0.3 --binary-search-steps 4 --verbose


# sigmaBinary_attack
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNN --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.2 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_rFGSM --batch-size 1024 --max-iterations 1000 --learning-rate 0.85 --threshold 0.6 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model AT_MaxMA --batch-size 1024 --max-iterations 1000 --learning-rate 4.0 --threshold 0.3 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DLA --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.4 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model DNNPlus --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.2 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model ICNN --batch-size 1024 --max-iterations 1000 --learning-rate 0.6 --threshold 0.4 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model PAD --batch-size 1024 --max-iterations 1000 --learning-rate 0.3 --threshold 0.3 --binary-search-steps 4 --verbose
python -m attacks.run_attacks.sigmaBinary_attack --cuda --data-path dataset/malscan_preprocessed --param-path defenses/saved_parameters --model KDE --batch-size 256 --max-iterations 1000 --learning-rate 0.5 --threshold 0.3 --binary-search-steps 4 --verbose

