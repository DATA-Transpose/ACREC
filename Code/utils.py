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

INCART = ['I01','I02','I03','I04','I05','I06','I07','I08',
          'I09','I10','I11','I12','I13','I14','I15','I16','I17','I18',
          'I19','I20','I21','I22','I23','I24','I25','I26','I27','I28','I29','I30','I31',
          'I32','I33','I34','I35','I36','I37','I38','I39','I40','I41','I42','I43','I44','I45',
          'I46','I47','I48','I49','I50','I51','I52','I53','I54','I55','I56','I57','I58','I59',
          'I60','I61','I62','I63','I64','I65','I66','I67','I68','I69','I70','I71','I72',
          'I73','I74','I75']