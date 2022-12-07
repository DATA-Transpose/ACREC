

path = '/content/drive/My Drive/DATA7902/save/'
train_data, train_label, test_data, test_label = LoadData(path, TRAIN, T2)

N_data, S_data, V_data, F_data = DivideClass(train_data, train_label)
#train_data, train_label = Balance_Data(N_data, S_data, V_data, F_data)
train_x, train_y = Test_Data(N_data, S_data, V_data, F_data)
train_loader = TrainLoader(train_x, train_y, 64, True)
N_data, S_data, V_data, F_data = DivideClass(test_data, test_label)
test_x, test_y = Test_Data(N_data, S_data, V_data, F_data)
test_loader = TestLoader(test_x, test_y, 64, True)

encoder_dir = '/content/drive/My Drive/DATA7902/Best_model/encoder_compare_nocausal.pt'
decoder_dir = '/content/drive/My Drive/DATA7902/Best_model/decoder_compare_nocausal.pt'
classifier_dir = '/content/drive/My Drive/DATA7902/Best_model/classifier_compare_nocausal.pt'
encoder=torch.load(encoder_dir)
decoder=torch.load(decoder_dir)
classifier=torch.load(classifier_dir)
test_records = TEST
path = '/content/drive/My Drive/DATA7902/save/'
data_dict, label_dict = LoadData3(path, [], TEST)
report = active_learning_loop2(encoder, classifier, test_records, data_dict, label_dict, mode = 'prob', q_size = 5, Epoch = 50)


encoder=torch.load(encoder_dir)
decoder=torch.load(decoder_dir)
classifier=torch.load(classifier_dir)
test_records = TRAIN + TEST
path = '/content/drive/My Drive/DATA7902/save/'
data_dict, label_dict = LoadData3(path, TRAIN, TEST)
report = active_learning_loop2(encoder, classifier, test_records, data_dict, label_dict, mode = 'prob', q_size = 5, Epoch = 30)