python src/run_model.py ~/scratch/NSNet/SATSolving/sr/test_hard --model BP --n_rounds 10
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 0 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 1 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 2 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 3 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 4 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 5 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 6 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 7 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 8 --model BP
python src/test_sat_solver.py ~/scratch/NSNet/SATSolving/sr/test_hard --solver Sparrow --max_flips 100 --n_process 32 --trial 9 --model BP

python src/show_sat_result.py runs/Sparrow/evaluations/ sr --model BP