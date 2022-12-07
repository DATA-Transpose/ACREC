import glob
import wfdb
import os

SAMPLE_RATES = {'MITDB': 360}
# Load raw signal data
class DataLoading():

    def __init__(self, load_path, dataset_name):
        self.load_path = load_path
        self.dataset_name = dataset_name
        self.fs = SAMPLE_RATES[dataset_name]

        self.records = [record.split('.')[0] for record in os.listdir(self.load_path)
                        if len(record.split('.')) > 1 and record.split('.')[1] == 'dat']

    # Load raw signal data and select the data in the first lead
    def load_dataset(self):
        print("Start Loading data")
        data_dict = {}
        signalfile = glob.glob(self.load_path + '*.hea')

        for i in range(len(signalfile)):
            annotation = wfdb.rdann(self.load_path + signalfile[i][-7:-4], 'atr')
            record_name = annotation.record_name
            print(record_name)
            signal = wfdb.rdsamp(self.load_path + record_name)[0][:, 0]
            label = annotation.symbol
            peak = annotation.sample
            data_dict[record_name] = {'signal': signal, 'category': label, 'peaks': peak, 'annotations': annotation}
            print("Records ", record_name, " has been loaded")
        return data_dict
