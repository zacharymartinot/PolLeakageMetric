import numpy as np, matplotlib.pyplot as plt
import glob
from leakage_metric import LeakageMetric

cst_dir = '/home/zmart/radcos/AaronSims/CTP/'
cst_files = glob.glob(cst_dir + '*.txt') # a list of strings where each string is the full path to a CST textfile

LM = LeakageMetric(cst_files, 'integral')

plt.plot(LM.freqs, LM.leakage_bound, c='k')
plt.show()
