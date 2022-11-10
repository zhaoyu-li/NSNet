python src/train_model.py sat-solving sat_neurosat_3-sat_marginal ~/scratch/NSNet/SATSolving/3-sat/train/ --valid_dir ~/scratch/NSNet/SATSolving/3-sat/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test/ --checkpoint runs/sat_neurosat_3-sat_marginal/checkpoints/model_best.pt --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test_hard/ --checkpoint runs/sat_neurosat_3-sat_marginal/checkpoints/model_best.pt --model NeuroSAT

python src/train_model.py sat-solving sat_neurosat_3-sat_assignment ~/scratch/NSNet/SATSolving/3-sat/train/ --valid_dir ~/scratch/NSNet/SATSolving/3-sat/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test/ --checkpoint runs/sat_neurosat_3-sat_assignment/checkpoints/model_best.pt --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test_hard/ --checkpoint runs/sat_neurosat_3-sat_assignment/checkpoints/model_best.pt --model NeuroSAT