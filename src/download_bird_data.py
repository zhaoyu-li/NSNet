import os
import wget
import tarfile
import glob
import argparse
import gzip
import shutil


url = 'https://www.comp.nus.edu.sg/~meel/Benchmarks/counting-cnfs.tar'


train_file_names = {
    'or_50': [
        'or-50-10-8-UC-30.cnf',
        'or-50-5-9-UC-30.cnf',
        'or-50-10-10-UC-40.cnf',
        'or-50-10-8-UC-40.cnf',
        'or-50-20-9-UC-40.cnf',
        'or-50-10-7-UC-30.cnf',
        'or-50-10-1-UC-40.cnf',
        'or-50-5-1-UC-20.cnf',
        'or-50-10-2-UC-40.cnf',
        'or-50-20-3-UC-40.cnf',
        'or-50-10-10-UC-20.cnf',
        'or-50-5-10-UC-40.cnf',
        'or-50-10-10-UC-30.cnf',
        'or-50-5-8-UC-30.cnf',
        'or-50-10-3-UC-30.cnf',
        'or-50-5-2-UC-40.cnf',
        'or-50-5-8-UC-40.cnf',
        'or-50-10-9-UC-30.cnf',
        'or-50-5-6-UC-30.cnf',
        'or-50-5-3-UC-40.cnf',
        'or-50-20-8-UC-40.cnf',
        'or-50-20-6-UC-40.cnf',
        'or-50-5-7-UC-40.cnf',
        'or-50-20-3-UC-30.cnf',
        'or-50-5-5-UC-30.cnf',
        'or-50-20-7-UC-40.cnf',
        'or-50-10-1-UC-20.cnf',
        'or-50-10-9-UC-40.cnf',
        'or-50-5-10-UC-20.cnf',
        'or-50-5-1-UC-30.cnf',
        'or-50-5-10-UC-30.cnf',
        'or-50-5-4-UC-40.cnf',
        'or-50-5-2-UC-30.cnf',
        'or-50-20-6-UC-30.cnf',
        'or-50-10-4-UC-40.cnf',
        'or-50-20-9-UC-30.cnf',
        'or-50-20-8-UC-30.cnf',
        'or-50-20-5-UC-40.cnf',
        'or-50-5-7-UC-30.cnf',
        'or-50-20-10-UC-40.cnf',
        'or-50-5-4-UC-30.cnf',
        'or-50-5-7-UC-20.cnf',
        'or-50-5-3-UC-10.cnf',
        'or-50-10-5-UC-40.cnf',
        'or-50-10-2-UC-10.cnf',
        'or-50-20-3-UC-20.cnf',
        'or-50-5-6-UC-10.cnf',
        'or-50-20-9-UC-20.cnf',
        'or-50-10-6-UC-20.cnf',
        'or-50-20-4-UC-20.cnf',
        'or-50-20-2-UC-30.cnf',
        'or-50-20-7-UC-20.cnf',
        'or-50-20-1-UC-30.cnf',
        'or-50-20-4-UC-30.cnf',
        'or-50-10-10-UC-10.cnf',
        'or-50-10-8-UC-10.cnf',
        'or-50-20-8-UC-20.cnf',
        'or-50-10-4-UC-20.cnf',
        'or-50-10-7-UC-20.cnf',
        'or-50-10-5-UC-30.cnf',
        'or-50-5-9-UC-20.cnf',
        'or-50-5-2-UC-20.cnf',
        'or-50-10-3-UC-20.cnf',
        'or-50-5-10-UC-10.cnf',
        'or-50-20-6-UC-20.cnf',
        'or-50-10-9-UC-20.cnf',
        'or-50-10-3-UC-10.cnf',
        'or-50-10-5-UC-20.cnf',
        'or-50-20-8-UC-10.cnf',
        'or-50-10-4-UC-10.cnf',
        'or-50-20-1-UC-20.cnf',
        'or-50-20-5-UC-20.cnf',
        'or-50-10-6-UC-10.cnf',
        'or-50-5-4-UC-20.cnf',
        'or-50-10-1-UC-10.cnf',
        'or-50-20-5-UC-10.cnf',
        'or-50-10-9-UC-10.cnf',
        'or-50-20-6-UC-10.cnf',
        'or-50-5-7-UC-10.cnf',
        'or-50-20-9-UC-10.cnf',
        'or-50-5-2-UC-10.cnf',
        'or-50-20-4.cnf',
        'or-50-5-5-UC-10.cnf',
        'or-50-20-3-UC-10.cnf',
        'or-50-10-7-UC-10.cnf',
        'or-50-20-3.cnf',
        'or-50-20-2.cnf',
        'or-50-5-3.cnf',
        'or-50-20-6.cnf',
        'or-50-5-5.cnf',
        'or-50-5-10.cnf',
        'or-50-5-1.cnf',
        'or-50-10-9.cnf',
        'or-50-20-8.cnf',
        'or-50-10-5.cnf',
        'or-50-5-6.cnf',
        'or-50-10-6.cnf',
        'or-50-5-4.cnf',
        'or-50-20-7.cnf',
        'or-50-20-9.cnf',
        'or-50-10-10.cnf',
        'or-50-20-5.cnf',
        'or-50-20-10.cnf',
        'or-50-20-1.cnf',
        'or-50-5-8.cnf',
    ],
    'or_60': [
        'or-60-5-9-UC-40.cnf',
        'or-60-10-6-UC-40.cnf',
        'or-60-20-2-UC-40.cnf',
        'or-60-5-1-UC-40.cnf',
        'or-60-5-4-UC-40.cnf',
        'or-60-5-8-UC-40.cnf',
        'or-60-20-9-UC-40.cnf',
        'or-60-5-2-UC-40.cnf',
        'or-60-5-7-UC-40.cnf',
        'or-60-10-5-UC-40.cnf',
        'or-60-10-9-UC-30.cnf',
        'or-60-10-9-UC-40.cnf',
        'or-60-5-4-UC-30.cnf',
        'or-60-5-8-UC-30.cnf',
        'or-60-10-4-UC-40.cnf',
        'or-60-10-7-UC-40.cnf',
        'or-60-10-8-UC-40.cnf',
        'or-60-20-8-UC-40.cnf',
        'or-60-5-9-UC-30.cnf',
        'or-60-10-8-UC-30.cnf',
        'or-60-5-6-UC-20.cnf',
        'or-60-5-10-UC-40.cnf',
        'or-60-5-5-UC-30.cnf',
        'or-60-10-10-UC-30.cnf',
        'or-60-5-4-UC-20.cnf',
        'or-60-10-10-UC-40.cnf',
        'or-60-20-2-UC-30.cnf',
        'or-60-5-10-UC-30.cnf',
        'or-60-20-7-UC-40.cnf',
        'or-60-5-6-UC-40.cnf',
        'or-60-5-7-UC-30.cnf',
        'or-60-20-9-UC-30.cnf',
        'or-60-10-2-UC-40.cnf',
        'or-60-20-1-UC-40.cnf',
        'or-60-20-9-UC-20.cnf',
        'or-60-10-4-UC-30.cnf',
        'or-60-10-9-UC-20.cnf',
        'or-60-5-9-UC-20.cnf',
        'or-60-5-2-UC-30.cnf',
        'or-60-10-5-UC-20.cnf',
        'or-60-10-7-UC-30.cnf',
        'or-60-10-6-UC-20.cnf',
        'or-60-20-1-UC-30.cnf',
        'or-60-20-8-UC-20.cnf',
        'or-60-10-7-UC-20.cnf',
        'or-60-5-2-UC-20.cnf',
        'or-60-20-3-UC-30.cnf',
        'or-60-10-8-UC-20.cnf',
        'or-60-20-5-UC-40.cnf',
        'or-60-10-10-UC-20.cnf',
        'or-60-5-4-UC-10.cnf',
        'or-60-10-1-UC-30.cnf',
        'or-60-5-10-UC-20.cnf',
        'or-60-20-3-UC-20.cnf',
        'or-60-20-10-UC-30.cnf',
        'or-60-5-7-UC-10.cnf',
        'or-60-20-4-UC-40.cnf',
        'or-60-5-1-UC-10.cnf',
        'or-60-10-3-UC-30.cnf',
        'or-60-20-2-UC-20.cnf',
        'or-60-5-10-UC-10.cnf',
        'or-60-20-4-UC-30.cnf',
        'or-60-10-9-UC-10.cnf',
        'or-60-5-3-UC-20.cnf',
        'or-60-10-5-UC-10.cnf',
        'or-60-5-2-UC-10.cnf',
        'or-60-5-5-UC-10.cnf',
        'or-60-10-4-UC-10.cnf',
        'or-60-10-1-UC-20.cnf',
        'or-60-20-5-UC-30.cnf',
        'or-60-5-6-UC-10.cnf',
        'or-60-20-9-UC-10.cnf',
        'or-60-20-4-UC-20.cnf',
        'or-60-20-8-UC-10.cnf',
        'or-60-20-6-UC-20.cnf',
        'or-60-20-7-UC-20.cnf',
        'or-60-5-8-UC-10.cnf',
        'or-60-10-10-UC-10.cnf',
        'or-60-10-1-UC-10.cnf',
        'or-60-20-1-UC-10.cnf',
        'or-60-10-6-UC-10.cnf',
        'or-60-20-4-UC-10.cnf',
        'or-60-20-6-UC-10.cnf',
        'or-60-20-5-UC-20.cnf',
        'or-60-5-7.cnf',
    ],
    'or_70': [
        'or-70-10-1-UC-40.cnf',
        'or-70-5-9-UC-30.cnf',
        'or-70-20-9-UC-40.cnf',
        'or-70-5-5-UC-30.cnf',
        'or-70-5-6-UC-40.cnf',
        'or-70-5-2-UC-40.cnf',
        'or-70-10-10-UC-40.cnf',
        'or-70-5-2-UC-30.cnf',
        'or-70-20-9-UC-30.cnf',
        'or-70-5-1-UC-30.cnf',
        'or-70-5-7-UC-40.cnf',
        'or-70-10-5-UC-40.cnf',
        'or-70-5-5-UC-40.cnf',
        'or-70-5-8-UC-30.cnf',
        'or-70-5-3-UC-20.cnf',
        'or-70-5-3-UC-30.cnf',
        'or-70-10-8-UC-40.cnf',
        'or-70-5-4-UC-40.cnf',
        'or-70-20-4-UC-40.cnf',
        'or-70-10-3-UC-30.cnf',
        'or-70-10-3-UC-40.cnf',
        'or-70-10-10-UC-30.cnf',
        'or-70-10-8-UC-30.cnf',
        'or-70-5-1-UC-40.cnf',
        'or-70-5-9-UC-20.cnf',
        'or-70-10-2-UC-30.cnf',
        'or-70-10-9-UC-40.cnf',
        'or-70-10-1-UC-30.cnf',
        'or-70-10-8-UC-20.cnf',
        'or-70-10-5-UC-30.cnf',
        'or-70-10-7-UC-40.cnf',
        'or-70-20-7-UC-40.cnf',
        'or-70-5-10-UC-30.cnf',
        'or-70-20-6-UC-30.cnf',
        'or-70-5-6-UC-30.cnf',
        'or-70-10-9-UC-30.cnf',
        'or-70-10-9-UC-20.cnf',
        'or-70-20-5-UC-40.cnf',
        'or-70-10-4-UC-40.cnf',
        'or-70-5-3-UC-10.cnf',
        'or-70-10-1-UC-20.cnf',
        'or-70-5-10-UC-20.cnf',
        'or-70-10-3-UC-20.cnf',
        'or-70-5-4-UC-30.cnf',
        'or-70-20-10-UC-40.cnf',
        'or-70-20-8-UC-40.cnf',
        'or-70-20-1-UC-40.cnf',
        'or-70-20-3-UC-40.cnf',
        'or-70-5-2-UC-20.cnf',
        'or-70-5-8-UC-20.cnf',
        'or-70-20-6-UC-20.cnf',
        'or-70-5-4-UC-20.cnf',
        'or-70-10-6-UC-20.cnf',
        'or-70-20-8-UC-30.cnf',
        'or-70-5-5-UC-20.cnf',
        'or-70-10-9-UC-10.cnf',
        'or-70-20-1-UC-30.cnf',
        'or-70-20-1-UC-20.cnf',
        'or-70-20-2-UC-30.cnf',
        'or-70-10-3-UC-10.cnf',
        'or-70-20-5-UC-30.cnf',
        'or-70-20-2-UC-20.cnf',
        'or-70-5-5-UC-10.cnf',
        'or-70-10-8-UC-10.cnf',
        'or-70-10-4-UC-30.cnf',
        'or-70-5-1-UC-10.cnf',
        'or-70-5-2-UC-10.cnf',
        'or-70-5-10-UC-10.cnf',
        'or-70-20-7-UC-20.cnf',
        'or-70-20-10-UC-30.cnf',
        'or-70-5-8-UC-10.cnf',
        'or-70-10-4-UC-20.cnf',
        'or-70-5-4-UC-10.cnf',
        'or-70-20-2-UC-10.cnf',
        'or-70-20-4-UC-20.cnf',
        'or-70-10-10-UC-10.cnf',
        'or-70-10-5-UC-10.cnf',
        'or-70-20-6-UC-10.cnf',
    ],
    'or_100': [
        'or-100-10-6-UC-60.cnf',
        'or-100-20-9-UC-50.cnf',
        'or-100-20-9-UC-60.cnf',
        'or-100-10-3-UC-60.cnf',
        'or-100-10-7-UC-60.cnf',
        'or-100-10-5-UC-60.cnf',
        'or-100-10-3-UC-40.cnf',
        'or-100-5-2-UC-60.cnf',
        'or-100-5-7-UC-40.cnf',
        'or-100-5-3-UC-40.cnf',
        'or-100-5-6-UC-40.cnf',
        'or-100-5-4-UC-60.cnf',
        'or-100-5-6-UC-50.cnf',
        'or-100-10-4-UC-60.cnf',
        'or-100-5-6-UC-60.cnf',
        'or-100-5-5-UC-60.cnf',
        'or-100-5-3-UC-50.cnf',
        'or-100-20-3-UC-50.cnf',
        'or-100-20-1-UC-60.cnf',
        'or-100-10-10-UC-60.cnf',
        'or-100-10-2-UC-40.cnf',
        'or-100-20-6-UC-60.cnf',
        'or-100-5-7-UC-60.cnf',
        'or-100-5-10-UC-40.cnf',
        'or-100-10-8-UC-60.cnf',
        'or-100-5-8-UC-60.cnf',
        'or-100-10-1-UC-40.cnf',
        'or-100-5-7-UC-50.cnf',
        'or-100-20-1-UC-50.cnf',
        'or-100-5-2-UC-50.cnf',
        'or-100-5-1-UC-30.cnf',
        'or-100-10-2-UC-60.cnf',
        'or-100-20-5-UC-60.cnf',
        'or-100-5-10-UC-60.cnf',
        'or-100-20-3-UC-60.cnf',
        'or-100-5-6-UC-30.cnf',
        'or-100-5-1-UC-60.cnf',
        'or-100-5-4-UC-50.cnf',
        'or-100-10-3-UC-50.cnf',
        'or-100-5-9-UC-40.cnf',
        'or-100-5-8-UC-40.cnf',
        'or-100-10-10-UC-50.cnf',
        'or-100-10-2-UC-50.cnf',
        'or-100-5-3-UC-30.cnf',
        'or-100-10-8-UC-50.cnf',
        'or-100-10-5-UC-50.cnf',
        'or-100-10-8-UC-40.cnf',
        'or-100-5-4-UC-40.cnf',
        'or-100-10-1-UC-30.cnf',
        'or-100-20-5-UC-50.cnf',
        'or-100-20-8-UC-60.cnf',
        'or-100-10-4-UC-40.cnf',
        'or-100-5-8-UC-30.cnf',
        'or-100-20-4-UC-60.cnf',
        'or-100-10-5-UC-40.cnf',
        'or-100-20-2-UC-40.cnf',
        'or-100-5-6-UC-20.cnf',
        'or-100-20-10-UC-60.cnf',
        'or-100-10-6-UC-30.cnf',
        'or-100-5-7-UC-30.cnf',
        'or-100-20-6-UC-50.cnf',
        'or-100-10-7-UC-40.cnf',
        'or-100-20-10-UC-50.cnf',
        'or-100-20-4-UC-50.cnf',
        'or-100-5-1-UC-20.cnf',
        'or-100-10-4-UC-30.cnf',
        'or-100-10-3-UC-30.cnf',
        'or-100-10-7-UC-30.cnf',
        'or-100-10-1-UC-20.cnf',
        'or-100-20-6-UC-40.cnf',
        'or-100-5-10-UC-20.cnf',
        'or-100-10-5-UC-30.cnf',
        'or-100-10-9-UC-40.cnf',
        'or-100-10-8-UC-30.cnf',
        'or-100-10-2-UC-30.cnf',
        'or-100-5-8-UC-20.cnf',
        'or-100-20-6-UC-30.cnf',
        'or-100-20-8-UC-40.cnf',
        'or-100-20-4-UC-40.cnf',
        'or-100-5-3-UC-10.cnf',
        'or-100-10-2-UC-20.cnf',
        'or-100-10-9-UC-30.cnf',
        'or-100-10-4-UC-20.cnf',
        'or-100-20-3-UC-40.cnf',
        'or-100-20-2-UC-60.cnf',
        'or-100-20-7-UC-40.cnf',
        'or-100-20-10-UC-30.cnf',
        'or-100-20-9-UC-30.cnf',
        'or-100-10-6-UC-20.cnf',
        'or-100-20-2-UC-30.cnf',
        'or-100-5-5-UC-20.cnf',
        'or-100-20-5-UC-30.cnf',
        'or-100-10-5-UC-20.cnf',
        'or-100-10-7-UC-20.cnf',
        'or-100-5-6-UC-10.cnf',
        'or-100-5-1-UC-10.cnf',
        'or-100-20-4-UC-20.cnf',
    ],
    's': [
        's27_15_7.cnf',
        's420_new_3_2.cnf',
        's820a_3_2.cnf',
        's298_3_2.cnf',
        's420_new1_3_2.cnf',
        's382_3_2.cnf',
        's349_3_2.cnf',
        's444_3_2.cnf',
        's382_7_4.cnf',
        's349_7_4.cnf',
        's510_3_2.cnf',
        's820a_7_4.cnf',
        's444_7_4.cnf',
        's510_15_7.cnf',
        's838_3_2.cnf',
        's420_7_4.cnf',
        's832a_7_4.cnf',
        's510_7_4.cnf',
        's820a_15_7.cnf',
        's838_7_4.cnf',
        's349_15_7.cnf',
        's420_new_15_7.cnf',
        's444_15_7.cnf',
        's344_15_7.cnf',
        's382_15_7.cnf',
        's832a_15_7.cnf',
        's1488_3_2.cnf',
        's526_3_2.cnf',
        's641_3_2.cnf',
        's953a_7_4.cnf',
        's838_15_7.cnf',
        's526a_3_2.cnf',
        's1488_15_7.cnf',
        's713_3_2.cnf',
        's298_15_7.cnf',
        's526a_7_4.cnf',
        's713_7_4.cnf',
        's35932_3_2.cnf',
        's713_15_7.cnf',
        's1238a_3_2.cnf',
        's1196a_7_4.cnf',
        's1238a_7_4.cnf',
        's526a_15_7.cnf',
        's526_15_7.cnf',
        's1238a_15_7.cnf',
        's1423a_3_2.cnf',
        's1423a_7_4.cnf',
        's1423a_15_7.cnf',
    ],
    'blasted': [
        'blasted_case200.cnf',
        'blasted_case103.cnf',
        'blasted_case134.cnf',
        'blasted_case30.cnf',
        'blasted_case64.cnf',
        'blasted_case127.cnf',
        'blasted_case100.cnf',
        'blasted_case206.cnf',
        'blasted_case33.cnf',
        'blasted_case28.cnf',
        'blasted_case24.cnf',
        'blasted_case59_1.cnf',
        'blasted_case58.cnf',
        'blasted_case27.cnf',
        'blasted_case17.cnf',
        'blasted_case102.cnf',
        'blasted_case23.cnf',
        'blasted_case25.cnf',
        'blasted_case59.cnf',
        'blasted_case36.cnf',
        'blasted_case136.cnf',
        'blasted_case47.cnf',
        'blasted_case124.cnf',
        'blasted_case55.cnf',
        'blasted_case11.cnf',
        'blasted_case112.cnf',
        'blasted_case53.cnf',
        'blasted_case22.cnf',
        'blasted_case52.cnf',
        'blasted_case43.cnf',
        'blasted_case4.cnf',
        'blasted_case51.cnf',
        'blasted_case201.cnf',
        'blasted_case205.cnf',
        'blasted_case203.cnf',
        'blasted_case1.cnf',
        'blasted_case204.cnf',
        'blasted_case5.cnf',
        'blasted_case110.cnf',
        'blasted_case111.cnf',
        'blasted_case38.cnf',
        'blasted_case113.cnf',
        'blasted_case44.cnf',
        'blasted_case117.cnf',
        'blasted_case54.cnf',
        'blasted_case135.cnf',
        'blasted_case_1_b14_3.cnf',
        'blasted_case_1_b14_2.cnf',
        'blasted_case6.cnf',
        'blasted_case68.cnf',
        'blasted_case_2_b14_2.cnf',
        'blasted_case108.cnf',
        'blasted_case_2_b14_3.cnf',
        'blasted_case_3_b14_3.cnf',
        'blasted_case126.cnf',
        'blasted_case131.cnf',
        'blasted_case109.cnf',
        'blasted_case_1_b14_1.cnf',
        'blasted_case123.cnf',
        'blasted_case2.cnf',
        'blasted_case_3_b14_1.cnf',
        'blasted_case121.cnf',
        'blasted_case125.cnf',
        'blasted_squaring22.cnf',
        'blasted_squaring24.cnf',
        'blasted_squaring23.cnf',
        'blasted_squaring21.cnf',
        'blasted_squaring20.cnf',
        'blasted_case119.cnf',
        'blasted_case115.cnf',
        'blasted_case120.cnf',
        'blasted_case211.cnf',
        'blasted_case143.cnf',
        'blasted_case39.cnf',
        'blasted_case40.cnf',
        'blasted_case34.cnf',
        'blasted_case213.cnf',
        'blasted_case214.cnf',
        'blasted_case207.cnf',
        'blasted_squaring25.cnf',
        'blasted_case130.cnf',
        'blasted_case_0_b12_1.cnf',
        'blasted_case_1_b12_1.cnf',
        'blasted_case50.cnf',
        'blasted_case212.cnf',
        'blasted_case209.cnf',
        'blasted_squaring29.cnf',
        'blasted_squaring51.cnf',
        'blasted_squaring30.cnf',
        'blasted_squaring28.cnf',
        'blasted_case105.cnf',
        'blasted_case145.cnf',
        'blasted_case146.cnf',
        'blasted_squaring1.cnf',
        'blasted_squaring3.cnf',
        'blasted_squaring4.cnf',
        'blasted_squaring2.cnf',
        'blasted_squaring5.cnf',
        'blasted_case49.cnf',
        'blasted_squaring8.cnf',
        'blasted_case_2_b12_2.cnf',
        'blasted_squaring9.cnf',
        'blasted_squaring7.cnf',
    ],
    '75': [
        '75-10-1-q.cnf',
        '75-10-10-q.cnf',
        '75-10-4-q.cnf',
        '75-10-7-q.cnf',
        '75-10-8-q.cnf',
        '75-10-2-q.cnf',
        '75-12-9-q.cnf',
        '75-12-4-q.cnf',
        '75-10-5-q.cnf',
        '75-10-9-q.cnf',
        '75-12-10-q.cnf',
        '75-12-7-q.cnf',
        '75-14-1-q.cnf',
        '75-12-3-q.cnf',
    ],
    '90': [
        '90-10-4-q.cnf',
        '90-10-10-q.cnf',
        '90-12-5-q.cnf',
        '90-10-9-q.cnf',
        '90-10-5-q.cnf',
        '90-12-10-q.cnf',
        '90-10-3-q.cnf',
        '90-12-1-q.cnf',
        '90-12-4-q.cnf',
        '90-16-3-q.cnf',
        '90-10-1-q.cnf',
        '90-12-6-q.cnf',
        '90-15-5-q.cnf',
        '90-14-2-q.cnf',
        '90-14-1-q.cnf',
        '90-15-7-q.cnf',
        '90-17-10-q.cnf',
        '90-14-6-q.cnf',
        '90-14-7-q.cnf',
        '90-15-4-q.cnf',
        '90-15-8-q.cnf',
        '90-14-9-q.cnf',
        '90-16-5-q.cnf',
        '90-18-9-q.cnf',
        '90-12-8-q.cnf',
        '90-20-6-q.cnf',
        '90-15-6-q.cnf',
        '90-15-10-q.cnf',
        '90-23-4-q.cnf',
        '90-17-9-q.cnf',
        '90-16-4-q.cnf',
        '90-20-1-q.cnf',
        '90-19-8-q.cnf',
        '90-24-8-q.cnf',
        '90-15-3-q.cnf',
        '90-16-6-q.cnf',
        '90-18-8-q.cnf',
        '90-16-7-q.cnf',
        '90-20-4-q.cnf',
        '90-21-3-q.cnf',
        '90-17-8-q.cnf',
        '90-18-7-q.cnf',
        '90-22-4-q.cnf',
        '90-16-9-q.cnf',
        '90-17-3-q.cnf',
        '90-21-5-q.cnf',
        '90-18-3-q.cnf',
        '90-20-8-q.cnf',
        '90-18-2-q.cnf',
        '90-21-1-q.cnf',
        '90-25-2-q.cnf',
        '90-22-6-q.cnf',
        '90-18-10-q.cnf',
        '90-14-5-q.cnf',
        '90-18-1-q.cnf',
        '90-24-2-q.cnf',
        '90-23-3-q.cnf',
        '90-20-9-q.cnf',
        '90-22-1-q.cnf',
        '90-20-7-q.cnf',
        '90-20-10-q.cnf',
        '90-20-3-q.cnf',
        '90-16-1-q.cnf',
        '90-22-3-q.cnf',
        '90-19-2-q.cnf',
        '90-19-1-q.cnf',
        '90-23-8-q.cnf',
        '90-23-6-q.cnf',
        '90-23-10-q.cnf',
        '90-17-4-q.cnf',
        '90-24-5-q.cnf',
        '90-19-7-q.cnf',
        '90-20-2-q.cnf',
        '90-25-1-q.cnf',
        '90-25-10-q.cnf',
    ]
}


test_file_names = {
    'or_50': [
        'or-50-5-3-UC-20.cnf',
        'or-50-5-5-UC-40.cnf',
        'or-50-5-6-UC-40.cnf',
        'or-50-5-1-UC-40.cnf',
        'or-50-10-6-UC-40.cnf',
        'or-50-10-3-UC-40.cnf',
        'or-50-10-2-UC-30.cnf',
        'or-50-10-7-UC-40.cnf',
        'or-50-5-3-UC-30.cnf',
        'or-50-5-9-UC-40.cnf',
        'or-50-5-6-UC-20.cnf',
        'or-50-10-1-UC-30.cnf',
        'or-50-20-1-UC-40.cnf',
        'or-50-10-8-UC-20.cnf',
        'or-50-10-4-UC-30.cnf',
        'or-50-10-2-UC-20.cnf',
        'or-50-20-4-UC-40.cnf',
        'or-50-20-7-UC-30.cnf',
        'or-50-20-5-UC-30.cnf',
        'or-50-20-2-UC-40.cnf',
        'or-50-20-10-UC-30.cnf',
        'or-50-5-8-UC-20.cnf',
        'or-50-10-6-UC-30.cnf',
        'or-50-5-5-UC-20.cnf',
        'or-50-20-4-UC-10.cnf',
        'or-50-20-10-UC-20.cnf',
        'or-50-10-5-UC-10.cnf',
        'or-50-5-9-UC-10.cnf',
        'or-50-20-7-UC-10.cnf',
        'or-50-5-1-UC-10.cnf',
        'or-50-20-2-UC-20.cnf',
        'or-50-20-10-UC-10.cnf',
        'or-50-5-8-UC-10.cnf',
        'or-50-5-4-UC-10.cnf',
        'or-50-10-2.cnf',
        'or-50-20-2-UC-10.cnf',
        'or-50-20-1-UC-10.cnf',
        'or-50-10-1.cnf',
        'or-50-10-7.cnf',
        'or-50-5-9.cnf',
        'or-50-10-3.cnf',
        'or-50-10-4.cnf',
        'or-50-10-8.cnf',
        'or-50-5-7.cnf',
        'or-50-5-2.cnf',
    ],
    'or_60': [
        'or-60-20-3-UC-40.cnf',
        'or-60-5-3-UC-40.cnf',
        'or-60-10-5-UC-30.cnf',
        'or-60-20-8-UC-30.cnf',
        'or-60-5-5-UC-40.cnf',
        'or-60-10-2-UC-30.cnf',
        'or-60-5-8-UC-20.cnf',
        'or-60-5-3-UC-30.cnf',
        'or-60-5-6-UC-30.cnf',
        'or-60-5-5-UC-20.cnf',
        'or-60-10-6-UC-30.cnf',
        'or-60-10-4-UC-20.cnf',
        'or-60-5-7-UC-20.cnf',
        'or-60-5-1-UC-30.cnf',
        'or-60-20-10-UC-40.cnf',
        'or-60-10-3-UC-40.cnf',
        'or-60-5-1-UC-20.cnf',
        'or-60-10-1-UC-40.cnf',
        'or-60-20-1-UC-20.cnf',
        'or-60-10-8-UC-10.cnf',
        'or-60-20-6-UC-40.cnf',
        'or-60-20-7-UC-30.cnf',
        'or-60-10-2-UC-20.cnf',
        'or-60-5-9-UC-10.cnf',
        'or-60-20-3-UC-10.cnf',
        'or-60-20-6-UC-30.cnf',
        'or-60-10-3-UC-20.cnf',
        'or-60-10-7-UC-10.cnf',
        'or-60-20-10-UC-20.cnf',
        'or-60-10-2-UC-10.cnf',
        'or-60-20-10-UC-10.cnf',
        'or-60-5-3-UC-10.cnf',
        'or-60-20-2-UC-10.cnf',
        'or-60-10-3-UC-10.cnf',
        'or-60-20-7-UC-10.cnf',
        'or-60-20-5-UC-10.cnf',
    ],
    'or_70': [
        'or-70-5-8-UC-40.cnf',
        'or-70-10-6-UC-40.cnf',
        'or-70-5-10-UC-40.cnf',
        'or-70-5-1-UC-20.cnf',
        'or-70-5-9-UC-40.cnf',
        'or-70-5-3-UC-40.cnf',
        'or-70-10-2-UC-40.cnf',
        'or-70-5-7-UC-30.cnf',
        'or-70-20-9-UC-20.cnf',
        'or-70-20-6-UC-40.cnf',
        'or-70-10-6-UC-30.cnf',
        'or-70-10-7-UC-30.cnf',
        'or-70-20-4-UC-30.cnf',
        'or-70-20-2-UC-40.cnf',
        'or-70-20-7-UC-30.cnf',
        'or-70-10-5-UC-20.cnf',
        'or-70-5-9-UC-10.cnf',
        'or-70-10-10-UC-20.cnf',
        'or-70-10-2-UC-20.cnf',
        'or-70-5-6-UC-20.cnf',
        'or-70-5-7-UC-20.cnf',
        'or-70-20-9-UC-10.cnf',
        'or-70-10-7-UC-20.cnf',
        'or-70-10-1-UC-10.cnf',
        'or-70-20-3-UC-30.cnf',
        'or-70-20-1-UC-10.cnf',
        'or-70-20-3-UC-20.cnf',
        'or-70-5-7-UC-10.cnf',
        'or-70-20-5-UC-20.cnf',
        'or-70-10-6-UC-10.cnf',
        'or-70-20-7-UC-10.cnf',
        'or-70-20-10-UC-10.cnf',
        'or-70-10-2-UC-10.cnf',
    ],
    'or_100': [
        'or-100-5-3-UC-60.cnf',
        'or-100-10-9-UC-60.cnf',
        'or-100-5-1-UC-40.cnf',
        'or-100-5-5-UC-50.cnf',
        'or-100-5-9-UC-30.cnf',
        'or-100-5-10-UC-50.cnf',
        'or-100-5-2-UC-40.cnf',
        'or-100-5-1-UC-50.cnf',
        'or-100-10-7-UC-50.cnf',
        'or-100-10-1-UC-60.cnf',
        'or-100-5-9-UC-50.cnf',
        'or-100-10-1-UC-50.cnf',
        'or-100-10-10-UC-40.cnf',
        'or-100-5-8-UC-50.cnf',
        'or-100-10-4-UC-50.cnf',
        'or-100-20-9-UC-40.cnf',
        'or-100-20-7-UC-60.cnf',
        'or-100-5-9-UC-60.cnf',
        'or-100-10-6-UC-50.cnf',
        'or-100-5-5-UC-40.cnf',
        'or-100-10-6-UC-40.cnf',
        'or-100-5-10-UC-30.cnf',
        'or-100-10-10-UC-30.cnf',
        'or-100-10-9-UC-50.cnf',
        'or-100-5-3-UC-20.cnf',
        'or-100-5-5-UC-30.cnf',
        'or-100-20-8-UC-50.cnf',
        'or-100-5-4-UC-30.cnf',
        'or-100-10-10-UC-20.cnf',
        'or-100-5-2-UC-30.cnf',
        'or-100-20-1-UC-40.cnf',
        'or-100-20-5-UC-40.cnf',
        'or-100-20-7-UC-50.cnf',
        'or-100-5-9-UC-20.cnf',
        'or-100-20-10-UC-40.cnf',
        'or-100-10-8-UC-20.cnf',
        'or-100-5-2-UC-20.cnf',
        'or-100-20-7-UC-30.cnf',
        'or-100-5-7-UC-20.cnf',
        'or-100-20-8-UC-30.cnf',
        'or-100-20-2-UC-50.cnf',
    ],
    's': [
        's27_new_3_2.cnf',
        's420_3_2.cnf',
        's832a_3_2.cnf',
        's344_3_2.cnf',
        's420_new1_7_4.cnf',
        's344_7_4.cnf',
        's420_new_7_4.cnf',
        's298_7_4.cnf',
        's420_15_7.cnf',
        's420_new1_15_7.cnf',
        's1488_7_4.cnf',
        's953a_15_7.cnf',
        's953a_3_2.cnf',
        's641_7_4.cnf',
        's526_7_4.cnf',
        's641_15_7.cnf',
        's35932_7_4.cnf',
        's1196a_3_2.cnf',
        's35932_15_7.cnf',
        's1196a_15_7.cnf',
    ],
    'blasted': [
        'blasted_case101.cnf',
        'blasted_case137.cnf',
        'blasted_case31.cnf',
        'blasted_case26.cnf',
        'blasted_case63.cnf',
        'blasted_case128.cnf',
        'blasted_case32.cnf',
        'blasted_case60.cnf',
        'blasted_case29.cnf',
        'blasted_case133.cnf',
        'blasted_case7.cnf',
        'blasted_case21.cnf',
        'blasted_case45.cnf',
        'blasted_case202.cnf',
        'blasted_case118.cnf',
        'blasted_case132.cnf',
        'blasted_case46.cnf',
        'blasted_case56.cnf',
        'blasted_case8.cnf',
        'blasted_case_3_b14_2.cnf',
        'blasted_case62.cnf',
        'blasted_case57.cnf',
        'blasted_case_2_b14_1.cnf',
        'blasted_case122.cnf',
        'blasted_case3.cnf',
        'blasted_case114.cnf',
        'blasted_case210.cnf',
        'blasted_case116.cnf',
        'blasted_squaring26.cnf',
        'blasted_case41.cnf',
        'blasted_case19.cnf',
        'blasted_squaring27.cnf',
        'blasted_case35.cnf',
        'blasted_case144.cnf',
        'blasted_case208.cnf',
        'blasted_case_2_b12_1.cnf',
        'blasted_squaring50.cnf',
        'blasted_case106.cnf',
        'blasted_squaring6.cnf',
        'blasted_squaring11.cnf',
        'blasted_case20.cnf',
        'blasted_case_1_b12_2.cnf',
        'blasted_squaring10.cnf',
        'blasted_case_0_b12_2.cnf',
    ],
    '75': [
        '75-10-6-q.cnf',
        '75-12-8-q.cnf',
        '75-12-5-q.cnf',
        '75-12-2-q.cnf',
        '75-14-5-q.cnf',
        '75-14-2-q.cnf',
    ],
    '90': [
        '90-12-2-q.cnf',
        '90-10-7-q.cnf',
        '90-16-2-q.cnf',
        '90-12-3-q.cnf',
        '90-14-3-q.cnf',
        '90-19-4-q.cnf',
        '90-17-1-q.cnf',
        '90-14-10-q.cnf',
        '90-14-8-q.cnf',
        '90-16-10-q.cnf',
        '90-18-4-q.cnf',
        '90-15-1-q.cnf',
        '90-17-5-q.cnf',
        '90-12-7-q.cnf',
        '90-17-6-q.cnf',
        '90-19-6-q.cnf',
        '90-22-9-q.cnf',
        '90-18-6-q.cnf',
        '90-20-5-q.cnf',
        '90-21-7-q.cnf',
        '90-19-10-q.cnf',
        '90-21-9-q.cnf',
        '90-26-5-q.cnf',
        '90-18-5-q.cnf',
        '90-24-10-q.cnf',
        '90-24-3-q.cnf',
        '90-15-2-q.cnf',
        '90-22-7-q.cnf',
        '90-24-9-q.cnf',
        '90-26-10-q.cnf',
        '90-23-5-q.cnf',
        '90-26-4-q.cnf',
    ]
}


def download(opts):
    file_name = os.path.basename(url)
    file_path = os.path.join(opts.out_dir, file_name)
    if not os.path.exists(file_path):
        wget.download(url, out=opts.out_dir)
    f = tarfile.open(file_path)
    f.extractall(opts.out_dir)
    f.close()
    os.remove(file_path)


def decompress(opts):
    all_files = sorted(glob.glob(opts.out_dir + '/**/*.gz', recursive=True))
    for f in all_files:
        file_name = os.path.basename(f)
        dir_name = os.path.dirname(f)
        out_file_name = file_name[:-15] # '.gz.no_w.cnf.gz'
        out_file_path = os.path.join(dir_name, out_file_name)

        with gzip.open(f, 'r') as f_in, open(out_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        os.remove(f)


def categorize(opts):
    category_map = {}
    split_filepath = os.path.join(opts.out_dir, 'train')
    for category in train_file_names:
        category_filepath = os.path.join(split_filepath, category)
        os.makedirs(category_filepath, exist_ok=True)
        for file_name in train_file_names[category]:
            category_map[file_name] = category_filepath
    
    split_filepath = os.path.join(opts.out_dir, 'test')
    for category in test_file_names:
        category_filepath = os.path.join(split_filepath, category)
        os.makedirs(category_filepath, exist_ok=True)
        for file_name in test_file_names[category]:
            category_map[file_name] = category_filepath
    
    all_files = sorted(glob.glob(opts.out_dir + '/**/*.cnf', recursive=True))
    for f in all_files:
        file_name = os.path.basename(f)
        if file_name in category_map:
            shutil.move(f, os.path.join(category_map[file_name], file_name))


def clean(opts):
    unused_dir = os.path.join(opts.out_dir, 'counting2')
    shutil.rmtree(unused_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    opts = parser.parse_args()

    os.makedirs(opts.out_dir, exist_ok=True)

    download(opts)
    decompress(opts)
    categorize(opts)
    clean(opts)


if __name__ == '__main__':
    main()
