python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/BIRD/test --solver F2 --timeout 5000
python src/test_mc_solver.py ~/scratch/NSNet/ModelCounting/BIRD_MIS/test --solver F2 --timeout 5000

python src/show_mc_result.py ~/scratch/NSNet/ModelCounting/BIRD/test/ runs/F2/evaluations/ runs/MIS/evaluations/