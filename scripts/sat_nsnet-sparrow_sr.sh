python src/run_model.py ~/scratch/NSNet/SATSolving/sr/test_hard --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt --model NSNet --n_rounds 10
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 0 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 1 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 2 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 3 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 4 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 5 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 6 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 7 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 8 --model NSNet
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 9 --model NSNet

python src/show_sat_result.py runs/Sparrow/evaluations/ sr --model NSNet