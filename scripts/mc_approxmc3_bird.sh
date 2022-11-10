python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/BIRD/test --solver ApproxMC3 --timeout 5000
python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/BIRD_MIS/test --solver ApproxMC3 --timeout 5000

python src/show_mc_result.py ~/scratch/NSNet/ModelCounting/BIRD/test/ runs/ApproxMC3/evaluations/ runs/MIS/evaluations/