python src/train_model.py sat-solving sat_nsnet_ca_marginal ~/scratch/NSNet/SATSolving/ca/train/ --valid_dir ~/scratch/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt

python src/train_model.py sat-solving sat_nsnet_ca_assignment ~/scratch/NSNet/SATSolving/ca/train/ --valid_dir ~/scratch/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_assignment/checkpoints/model_best.pt
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_assignment/checkpoints/model_best.pt