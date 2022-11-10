python src/train_model.py sat-solving sat_nsnet_sr_marginal ~/scratch/NSNet/SATSolving/sr/train/ --valid_dir ~/scratch/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt

python src/train_model.py sat-solving sat_nsnet_sr_assignment ~/scratch/NSNet/SATSolving/sr/train/ --valid_dir ~/scratch/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_nsnet_sr_assignment/checkpoints/model_best.pt
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_nsnet_sr_assignment/checkpoints/model_best.pt