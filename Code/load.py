from utils import *


def LoadData(path, train_records, test_records):
    train_data = []
    train_label = []
    for record in train_records:
        data_file = record + 'Data.npy'
        label_file = record + 'Label.npy'
        data = np.load(path + data_file, allow_pickle=True)
        data.astype(np.float32)
        label = np.load(path + label_file, allow_pickle=True)
        train_data += list(data)
        train_label += list(label)

    test_data = []
    test_label = []
    for record in test_records:
        data_file = record + 'Data.npy'
        label_file = record + 'Label.npy'
        data = np.load(path + data_file, allow_pickle=True)
        data.astype(np.float32)
        label = np.load(path + label_file, allow_pickle=True)
        test_data += list(data)
        test_label += list(label)

    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)


def LoadData2(path, train_records, test_records):
    train_data = []
    train_label = []
    for record in train_records + test_records:
        data_file = record + 'Data.npy'
        label_file = record + 'Label.npy'
        data = np.load(path + data_file, allow_pickle=True)
        data.astype(np.float32)
        label = np.load(path + label_file, allow_pickle=True)
        train_data += list(data)
        train_label += list(label)

    return np.array(train_data), np.array(train_label)


def LoadData3(path, train_records, test_records):
    data_dict = {}
    label_dict = {}
    for record in train_records + test_records:
        data_file = record + 'Data.npy'
        label_file = record + 'Label.npy'
        data = np.load(path + data_file, allow_pickle=True)
        data.astype(np.float32)
        label = np.load(path + label_file, allow_pickle=True)
        data_dict[record] = list(data)
        label_dict[record] = list(label)

    return data_dict, label_dict


def DivideClass(data, label):
    N_data = []
    S_data = []
    V_data = []
    F_data = []
    for i in range(len(data)):
        if label[i] in AAMI[0]:
            N_data.append(data[i])
        elif label[i] in AAMI[1]:
            S_data.append(data[i])
        elif label[i] in AAMI[2]:
            V_data.append(data[i])
        elif label[i] in AAMI[3]:
            F_data.append(data[i])

    print('N: ', len(N_data))
    print('S: ', len(S_data))
    print('V: ', len(V_data))
    print('F: ', len(F_data))
    return np.array(N_data), np.array(S_data), np.array(V_data), np.array(F_data)


def Test_Data(N_data, S_data, V_data, F_data):
    n_sample = list(N_data)
    n_label = [0 for i in range(len(N_data))]

    s_sample = list(S_data)
    s_label = [1 for i in range(len(S_data))]

    v_sample = list(V_data)
    v_label = [2 for i in range(len(V_data))]

    f_sample = list(F_data)
    f_label = [3 for i in range(len(F_data))]
    test_data = n_sample + s_sample + v_sample + f_sample
    test_label = n_label + s_label + v_label + f_label
    # test_data = [test_data[i].reshape((73,73)) for i in range(len(test_data))]
    return np.array(test_data), np.array(test_label)