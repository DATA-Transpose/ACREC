from utils import *
import scipy.signal as ss
import dtcwt
import numpy as np
from sklearn import preprocessing
import signal_load

# Preprocess raw signal data and save them into npy files
class DataPreprocess():

    def __init__(self, data_dict, records, fs, save_path):

        self.data_dict = data_dict
        self.save_path = save_path
        self.fs = fs
        self.records = records
        self.avg_len = {}

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # Denoise signals by dtcwt
    def dtcwt_denoise(self):

        for record in self.records:
            print(record + ' are denoised.')
            dtcwt_base = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
            self.coeffs = dtcwt_base.forward(self.data_dict[record]['signal'], nlevels=11, include_scale=True)
            denoised_signal = dtcwt_base.inverse(self.coeffs, gain_mask=[0] + [1 for i in range(8)] + [0, 0])

            self.data_dict[record]['signal'] = denoised_signal

    # Median filter signals
    def median_fit(self):
        for record in self.records:
            print(record + ' are filtered.')
            filter_1 = ss.medfilt(self.data_dict[record]['signal'], int(360 * 200 / 1000 + 1))
            filter_2 = ss.medfilt(filter_1, int(360 * 600 / 1000 + 1))

            self.data_dict[record]['signal'] = np.array(self.data_dict[record]['signal']) - filter_2

    # Segment signal in a fixed length
    def segmentation(self, normalization=False):

        for record in self.records:
            beat = []
            for i in range(10, len(self.data_dict[record]['peaks']) - 10):
                peak = self.data_dict[record]['peaks'][i]
                single_beat = self.data_dict[record]['signal'][peak - 120:peak + 180]
                if normalization:
                    single_beat = self.Nomalization(single_beat.reshape(-1, 1))

                beat.append(single_beat)
            self.data_dict[record]['heartbeat'] = beat
            self.data_dict[record]['category'] = self.data_dict[record]['category'][
                                                 10:len(self.data_dict[record]['peaks']) - 10]

            print(len(self.data_dict[record]['heartbeat']))
            print(len(self.data_dict[record]['category']))

    # Segment signal in dynamic lengths
    def segmentation_dynamic(self, normalization=False):

        self.MeanLength()
        for record in self.records:
            beat = []
            category = []
            print(record + ' are segmentated.')
            for i in range(10, len(self.data_dict[record]['peaks']) - 10):
                peak = self.data_dict[record]['peaks'][i]
                len_ = self.avg_len[record][i - 10]
                before = int(len_ * 0.9)
                after = int(len_ * 0.6)
                single_beat = self.data_dict[record]['signal'][peak - before:peak + after]
                single_beat = ss.resample(single_beat, 300, axis=0)
                if normalization:
                    single_beat = self.Nomalization(single_beat.reshape(-1, 1))

                beat.append(single_beat)

            self.data_dict[record]['heartbeat'] = beat
            self.data_dict[record]['category'] = self.data_dict[record]['category'][
                                                 10:len(self.data_dict[record]['peaks']) - 10]

            print(len(self.data_dict[record]['heartbeat']))
            print(len(self.data_dict[record]['category']))

    # convert the category of heartbeat to AAMI standards
    def segmentation_category(self, category_list):

        class_list = [0, 0, 0, 0, 0]
        for label in category_list:
            if label in AAMI[0]:
                class_list[0] += 1
            elif label in AAMI[1]:
                class_list[1] += 1
            elif label in AAMI[2]:
                class_list[2] += 1
            elif label in AAMI[3]:
                class_list[3] += 1
            else:
                class_list[4] += 1
        max_index = np.argmax(np.array(class_list[1:]))
        if sum(class_list[1:]) == 0:
            return 'N'
        else:
            print(class_list)
        if max_index == 0:
            return 'S'
        elif max_index == 1:
            return 'V'
        elif max_index == 2:
            return 'F'
        elif max_index == 3:
            return 'Q'

    # normalize signals by StandardScaler
    def Nomalization(self, sig):

        scaler_1 = preprocessing.StandardScaler()

        sig_2 = scaler_1.fit_transform(sig)
        return sig_2.reshape(1, -1)

    # calculate mean length of the closed 20 signals
    def MeanLength(self):
        for record in self.records:
            self.avg_len[record] = []
            for i in range(10, len(self.data_dict[record]['peaks']) - 10):
                count = 20
                length = self.data_dict[record]['peaks'][i + 10] - self.data_dict[record]['peaks'][i - 10]
                avg_len = int(length / count)
                self.avg_len[record].append(avg_len)

    # save signal data into npy file
    def save_data(self):
        for record in self.records:
            np.save(self.save_path + record + 'Data', self.data_dict[record]['heartbeat'])
            np.save(self.save_path + record + 'Label', self.data_dict[record]['category'])

# Example code for pre-processing
# load_path = './Data/mitdb/'
# save_path = './Data/save/'
# a = signal_load.DataLoading(load_path, 'MITDB')
#
# dict_ = a.load_dataset()
# b = DataPreprocess(dict_, a.records, a.fs, save_path)
# b.median_fit()
# b.segmentation_dynamic(True)
# b.save_data()