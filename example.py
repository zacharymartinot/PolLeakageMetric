import numpy as np, matplotlib.pyplot as plt
import glob
from leakage_metric import LeakageMetric

cst_dir = cst_dir = '/home/zmart/radcos/AaronSims/CTP/'
cst_files = glob.glob(cst_dir + '*.txt') # a list of strings where each string is the full path to a CST textfile

LM_n = LeakageMetric(cst_files, 'none')

plt.plot(LM_n.freqs, LM_n.leakage_bound, c='k')
plt.show()
