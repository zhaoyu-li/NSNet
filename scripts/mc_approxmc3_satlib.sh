python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/SATLIB/test --solver ApproxMC3 --timeout 5000
python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/SATLIB_MIS/test --solver ApproxMC3 --timeout 5000

python src/show_mc_result.py ~/scratch/NSNet/ModelCounting/SATLIB/test/ runs/ApproxMC3/evaluations/ runs/MIS/evaluations/