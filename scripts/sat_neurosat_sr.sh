python src/train_model.py sat-solving sat_neurosat_sr_marginal ~/scratch/NSNet/SATSolving/sr/train/ --valid_dir ~/scratch/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_neurosat_sr_marginal/checkpoints/model_best.pt --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_neurosat_sr_marginal/checkpoints/model_best.pt --model NeuroSAT

python src/train_model.py sat-solving sat_neurosat_sr_assignment ~/scratch/NSNet/SATSolving/sr/train/ --valid_dir ~/scratch/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_neurosat_sr_assignment/checkpoints/model_best.pt --model NeuroSAT
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_neurosat_sr_assignment/checkpoints/model_best.pt --model NeuroSAT