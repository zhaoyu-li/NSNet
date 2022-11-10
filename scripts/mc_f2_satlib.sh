python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/SATLIB/test --solver F2 --timeout 5000
python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/SATLIB_MIS/test --solver F2 --timeout 5000

python src/show_mc_result.py ~/scratch/NSNet/ModelCounting/SATLIB/test/ runs/F2/evaluations/ runs/MIS/evaluations/