from utils import *
import train as tr
import sys
import load
import torch
import model
import train

def MAIN():
    random_setting()

    path = './save/'
    encoder_dir = './Best_model/encoder.pt'
    decoder_dir = './Best_model/decoder.pt'
    classifier_dir = './Best_model/classifier.pt'
    train_data, train_label = load.LoadData(path, TRAIN)

    N_data, S_data, V_data, F_data = load.DivideClass(train_data, train_label)
    train_x, train_y = load.Test_Data(N_data, S_data, V_data, F_data)
    train_loader = train.TrainLoader(train_x, train_y, 64, True)

    if Stage == 'Pretrain':
        encoder = model.LSTM_CNNEncoder()
        decoder = model.LSTM_CNNDecoder()
        classifier = model.Classifier()
        encoder, decoder, classifier = train.Pretrain_Causal(encoder, decoder, classifier, train_loader, GPU_device=True,
                                                             alpha=0.95, beta=0.05, Epoch=100, causal=True)

        torch.save(encoder, encoder_dir)
        torch.save(decoder, decoder_dir)
        torch.save(classifier, classifier_dir)
    else:

        encoder=torch.load(encoder_dir)
        classifier=torch.load(classifier_dir)
        if Dataset_Names == 'TEST':
            test_records = TEST
        else:
            test_records = SVDB

        data_dict, label_dict = load.LoadDataDict(path, TEST)
        report = train.active_learning_loop(encoder, classifier, test_records, data_dict, label_dict, mode = 'prob', q_size = 5, Epoch = 50)
        print(report)

if __name__ == "__main__":

    # Read settings
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--Dataset_Name':
            Dataset_Names = sys.argv[i + 1]
        if sys.argv[i] == '--Stage' and i + 1 < len(sys.argv):
            Stage = sys.argv[i + 1]

    MAIN()
