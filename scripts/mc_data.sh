# bird data
python src/download_bird_data.py ~/scratch/NSNet/ModelCounting/BIRD
python src/clean_data.py ~/scratch/NSNet/ModelCounting/BIRD
python src/generate_labels.py model-counting ~/scratch/NSNet/ModelCounting/BIRD/train
python src/generate_labels.py model-counting ~/scratch/NSNet/ModelCounting/BIRD/test

# satlib data
python src/download_satlib_data.py ~/scratch/NSNet/ModelCounting/SATLIB
python src/clean_data.py ~/scratch/NSNet/ModelCounting/SATLIB
python src/generate_labels.py model-counting ~/scratch/NSNet/ModelCounting/SATLIB
python src/split_satlib_data.py ~/scratch/NSNet/ModelCounting/SATLIB --keep_category

# mis preprocessing
python src/run_mis_solver.py ~/scratch/NSNet/ModelCounting/BIRD/test ~/scratch/NSNet/ModelCounting/BIRD_MIS/test --timeout 1000
python src/run_mis_solver.py ~/scratch/NSNet/ModelCounting/SATLIB/test ~/scratch/NSNet/ModelCounting/SATLIB_MIS/test --timeout 1000
