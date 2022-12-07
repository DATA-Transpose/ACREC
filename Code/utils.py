import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.modules import dropout
import pickle
import sklearn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import glob
from biosppy.signals import ecg
import pywt
import scipy.signal as ss
import dtcwt
from scipy.interpolate import interp1d

import wfdb
import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.data as data_utils
from sklearn.preprocessing import scale
from scipy.signal import resample
import os
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

AAMI = [['.','N','L','R','e','j'],['A','a','J','S'],['V','E'],['F'],['/','f','Q','!']]

TRAIN = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203','205', '207', '208', '209', '215', '220',
       '223', '230']

TEST = ['100', '103', '105', '111', '113', '117', '121', '123', '200',
            '202', '210', '212', '213', '214', '219', '221', '222', '228',
            '231', '232', '233', '234']

SVDB = ['800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810',
        '811', '812', '820', '821', '822', '823', '824', '825', '826', '827', '828',
        '829', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849',
        '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860',
        '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872',
        '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884',
        '885', '886', '887', '888', '889', '890', '891', '892', '893', '894']


Dataset_Names = 'TEST'
# 'TEST', 'SVDB'
Stage = 'Pretrain'
# 'Pretrain', 'Active'

def random_setting(seed = 42):
    # random seed setting
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True