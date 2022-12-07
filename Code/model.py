from utils import *


class LSTM_CNNEncoder(nn.Module):
    def __init__(self, K=16):
        super(LSTM_CNNEncoder, self).__init__()
        self.rnn = nn.LSTM(1, 1184, 1, batch_first=True)
        self.drop1 = nn.Dropout(0.4)
        self.conv1 = nn.Conv1d(1, 4, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(4)
        self.pooling1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.pooling2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1184, K)

    def forward(self, x):
        # print('1: ', x.shape)
        batch_size = x.size(0)
        x = x.squeeze(1)
        x = x.unsqueeze(2)
        # print('2: ', x.shape)
        out_rnn, (h_n, h_c) = self.rnn(x, None)
        out_rnn = self.drop1(out_rnn[:, -1, :])
        # print('3: ', out_rnn.shape)
        x = x.squeeze(2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # print('4: ', x.shape)
        x = F.relu(self.pooling1(self.bn1(x)))
        x = self.conv2(x)
        x = F.relu(self.pooling2(self.bn2(x)))
        x = self.drop2(x)
        out_cnn = x.view(batch_size, -1)
        batch_size = x.size(0)
        x = out_rnn + out_cnn
        out = self.fc1(x)
        # print('out: ', out.shape)
        return out


class LSTM_CNNDecoder(nn.Module):
    def __init__(self, K=16):
        super(LSTM_CNNDecoder, self).__init__()
        self.fc1 = nn.Linear(K, 1184)
        self.dconv1 = nn.ConvTranspose1d(16, 4, kernel_size=3, stride=2, padding=0)
        self.dconv2 = nn.ConvTranspose1d(4, 1, kernel_size=3, stride=2, padding=0, output_padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        # print('1: ', x.shape)
        x = F.relu(self.fc1(F.relu(x)))
        # print('2: ', x.shape)
        x = x.view(batch_size, 16, -1)
        # print('3: ', x.shape)
        x = F.relu(self.dconv1(x))
        # print('4: ', x.shape)
        # print('5: ', x.shape)
        out = self.dconv2(x)
        # print('6: ', out.shape)

        return out


class Classifier(nn.Module):
    def __init__(self, K=16):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(K, 8)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(F.relu(x)))
        x = self.drop(x)
        out = self.fc2(x)
        return out
