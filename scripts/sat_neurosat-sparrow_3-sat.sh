python src/run_model.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --checkpoint runs/sat_neurosat_3-sat_marginal/checkpoints/model_best.pt --model NeuroSAT --n_rounds 10
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 0 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 1 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 2 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 3 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 4 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 5 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 6 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 7 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 8 --model NeuroSAT
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 9 --model NeuroSAT

python src/show_sat_result.py runs/Sparrow/evaluations/ 3-sat --model NeuroSAT