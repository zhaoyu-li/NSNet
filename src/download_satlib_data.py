import os
import wget
import tarfile
import argparse

urls = {
    'RND3SAT':
    [
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf75-325.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf100-430.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf125-538.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf150-645.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf175-753.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf200-860.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf225-960.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz',
    ],
    'BMS':
    [
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/BMS/RTI_k3_n100_m429.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/BMS/BMS_k3_n100_m429.tar.gz',
    ],
    'CBS':
    [
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m411_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m411_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m411_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m411_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m411_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m418_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m418_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m418_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m418_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m418_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m423_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m423_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m423_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m423_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m423_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m429_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m429_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m429_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m429_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m429_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m435_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m435_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m441_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m435_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m435_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m441_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m441_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m441_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m441_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m441_b90.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m449_b10.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m449_b30.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m449_b50.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m449_b70.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m449_b90.tar.gz',
    ],
    'GCP':
    [
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat30-60.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat50-115.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat75-180.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat100-239.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat125-301.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat150-360.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat175-417.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat200-479.tar.gz',
    ],
    'SW-GCP':
    [
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp0-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp1-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp2-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp3-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp4-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp5-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp6-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp7-c5.tar.gz',
        'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp8-c5.tar.gz',
    ]
}

def download(opts):
    for category in urls:
        category_path = os.path.join(opts.out_dir, category)
        for url in urls[category]:
            file_name = os.path.basename(url)
            dir_name = file_name.split('.')[0]
            dir_path = os.path.join(category_path, dir_name)
            file_path = os.path.join(dir_path, file_name)
            os.makedirs(dir_path, exist_ok=True)
            if not os.path.exists(file_path):
                wget.download(url, out=dir_path)
            f = tarfile.open(file_path)
            f.extractall(dir_path)
            f.close()
            os.remove(file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    opts = parser.parse_args()

    os.makedirs(opts.out_dir, exist_ok=True)

    download(opts)


if __name__ == '__main__':
    main()
