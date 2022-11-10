python src/train_model.py sat-solving sat_nsnet_3-sat_marginal ~/scratch/NSNet/SATSolving/3-sat/train/ --valid_dir ~/scratch/NSNet/SATSolving/3-sat/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test/ --checkpoint runs/sat_nsnet_3-sat_marginal/checkpoints/model_best.pt
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test_hard/ --checkpoint runs/sat_nsnet_3-sat_marginal/checkpoints/model_best.pt

python src/train_model.py sat-solving sat_nsnet_3-sat_assignment ~/scratch/NSNet/SATSolving/3-sat/train/ --valid_dir ~/scratch/NSNet/SATSolving/3-sat/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test/ --checkpoint runs/sat_nsnet_3-sat_assignment/checkpoints/model_best.pt
python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/3-sat/test_hard/ --checkpoint runs/sat_nsnet_3-sat_assignment/checkpoints/model_best.pt