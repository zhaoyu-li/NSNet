# sr data
python src/generate_sr_data.py ~/scratch/NSNet/SATSolving/sr/train 30000 --min_n 10 --max_n 40
python src/generate_sr_data.py ~/scratch/NSNet/SATSolving/sr/valid 10000 --min_n 10 --max_n 40
python src/generate_sr_data.py ~/scratch/NSNet/SATSolving/sr/test 10000 --min_n 10 --max_n 40
python src/generate_sr_data.py ~/scratch/NSNet/SATSolving/sr/test_hard 10000 --min_n 40 --max_n 200

python src/generate_labels.py marginal ~/scratch/NSNet/SATSolving/sr/train
python src/generate_labels.py marginal ~/scratch/NSNet/SATSolving/sr/valid

python src/generate_labels.py assignment ~/scratch/NSNet/SATSolving/sr/train
python src/generate_labels.py assignment ~/scratch/NSNet/SATSolving/sr/valid

# 3-sat data
python src/generate_3-sat_data.py ~/scratch/NSNet/SATSolving/3-sat/train 30000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py ~/scratch/NSNet/SATSolving/3-sat/valid 10000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py ~/scratch/NSNet/SATSolving/3-sat/test 10000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py ~/scratch/NSNet/SATSolving/3-sat/test_hard 10000 --min_n 40 --max_n 200

python src/generate_labels.py marginal ~/scratch/NSNet/SATSolving/3-sat/train
python src/generate_labels.py marginal ~/scratch/NSNet/SATSolving/3-sat/valid

python src/generate_labels.py assignment ~/scratch/NSNet/SATSolving/3-sat/train
python src/generate_labels.py assignment ~/scratch/NSNet/SATSolving/3-sat/valid

# ca data
python src/generate_ca_data.py ~/scratch/NSNet/SATSolving/ca/train 30000 --min_n 10 --max_n 40
python src/generate_ca_data.py ~/scratch/NSNet/SATSolving/ca/valid 10000 --min_n 10 --max_n 40
python src/generate_ca_data.py ~/scratch/NSNet/SATSolving/ca/test 10000 --min_n 10 --max_n 40
python src/generate_ca_data.py ~/scratch/NSNet/SATSolving/ca/test_hard 10000 --min_n 40 --max_n 200

python src/generate_labels.py marginal ~/scratch/NSNet/SATSolving/ca/train
python src/generate_labels.py marginal ~/scratch/NSNet/SATSolving/ca/valid

python src/generate_labels.py assignment ~/scratch/NSNet/SATSolving/ca/train
python src/generate_labels.py assignment ~/scratch/NSNet/SATSolving/ca/valid