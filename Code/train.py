from utils import *

def TrainLoader(train_data, train_label, batch_size, shuffle):
    #trainset = TrainDataset(train_data, train_label)
    tensor_train_data = torch.Tensor(train_data)
    tensor_train_labels = torch.Tensor(np.array(train_label)).long()
    trainset = data_utils.TensorDataset(tensor_train_data, tensor_train_labels)
    train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return train_loader

def TestLoader(test_data, test_label, batch_size, shuffle):
    #testset = TestDataset(test_data, test_label)
    tensor_test_data = torch.Tensor(test_data)
    tensor_test_labels = torch.Tensor(np.array(test_label)).long()
    testset = data_utils.TensorDataset(tensor_test_data, tensor_test_labels)
    test_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return test_loader


def classification_loss(outputs, target, weights=None):
    class_weights = np.exp(-1 * np.array([45490.0, 929.0, 3770.0, 412.0]) / 75000) / np.sum(
        np.exp(-1 * np.array([45490.0, 929.0, 3770.0, 412.0]) / 75000))
    class_weights = torch.FloatTensor(class_weights).cuda()
    if weights != None:
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        CE = criterion(outputs, target)
        sample_weights = weights
        CE = torch.sum((CE * sample_weights / torch.sum(sample_weights)))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        CE = criterion(outputs, target)
    return CE


def balance_loss(weights, latent):
    loss = 0
    latent = (torch.sign(latent) + 1) / 2
    for j in range(latent.size(1)):
        x_j_1 = latent[:, j].unsqueeze(1)
        x_j_0 = (1 - latent[:, j]).unsqueeze(1)
        x_j_n = latent.clone().detach()
        x_j_n[:, j] = 0
        # print(x_j_n.shape)
        a = x_j_0
        # print(a.shape)
        loss += torch.norm(
            x_j_n.t().mm(weights * x_j_1) / (weights.t().mm(x_j_1)[0] + 1) - x_j_n.t().mm(weights * x_j_0) / (
                        weights.t().mm(x_j_0)[0] + 1)) ** 2

    loss += (torch.sum(weights) - 1) ** 2

    return loss


def Pretrain_Causal(encoder, decoder, classifier, train_loader, lr=0.001, GPU_device=True, Epoch=40, alpha=1, beta=1,
                    gamma=1, causal=True):
    weights_list = []
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        shape = torch.empty(target.size(0), 1)
        weights = torch.ones_like(shape)
        weights = torch.FloatTensor(weights) / target.size(0)
        weights = weights.cuda()
        weights_list.append(weights)
    # print(len(weights_list))

    optimizer2 = torch.optim.NAdam(classifier.parameters(), 0.0005, betas=(0.9, 0.999), eps=1e-07)
    optimizer3 = torch.optim.NAdam(encoder.parameters(), 0.0005, betas=(0.9, 0.999), eps=1e-07)
    optimizer4 = torch.optim.NAdam(decoder.parameters(), 0.0001, betas=(0.9, 0.999), eps=1e-07)
    lossf = nn.MSELoss(reduction='mean')

    running_loss = 0.0
    # model_dir = '/content/drive/My Drive/DATA7902/Best_model/causal_model.pt'

    for epoch in range(Epoch):

        sum_loss1 = 0
        sum_loss2 = 0
        sum_loss3 = 0
        # print('Epoch: ', epoch)
        count = 0
        encoder.train()
        decoder.train()
        classifier.train()
        for batch_idx, data in enumerate(train_loader):
            # print(batch_idx)
            weights = weights_list[batch_idx]
            # print(weights.shape)
            optimizer1 = torch.optim.NAdam([weights], lr, betas=(0.9, 0.999), eps=1e-07)

            inputs, target = data
            size = target.size(0)
            count += size
            if GPU_device:
                inputs = inputs.cuda()
                target = target.cuda()
                encoder.cuda()
                decoder.cuda()
                classifier.cuda()
                # weights.cuda()
            # print(inputs.shape)
            latent = encoder(inputs)
            re_construct = decoder(latent)
            if causal == True:
                latent = weights * latent
            outputs = classifier(latent)

            if causal == True:
                loss1 = balance_loss(weights, latent)
                optimizer1.zero_grad()
                loss1.backward(retain_graph=True)
                optimizer1.step()

            if causal == True:
                loss2 = alpha * classification_loss(outputs, target, weights)  # + beta * lossf(re_construct, inputs)
            else:
                loss2 = alpha * classification_loss(outputs, target)  # + beta * lossf(re_construct, inputs)
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            # optimizer4.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer2.step()
            optimizer3.step()
            # optimizer4.step()

            # print(re_construct.shape)
            if causal == True:
                sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
            # sum_loss3 += loss3.item()
        # if causal == True:
        # print('loss1: ', sum_loss1/batch_idx)
        # print('loss2: ', sum_loss2/batch_idx)
    # torch.save(model, model_dir)
    return encoder, decoder, classifier


def active_learning_loop2(encoder, classifier, test_records, data_dict, label_dict, mode='prob', q_size=50, lr=0.0005,
                          GPU_device=True, Epoch=20):
    optimizer1 = torch.optim.NAdam(classifier.parameters(), lr, betas=(0.9, 0.999), eps=1e-07)
    optimizer2 = torch.optim.NAdam(encoder.parameters(), lr, betas=(0.9, 0.999), eps=1e-07)
    running_loss = 0.0
    min_loss = 100000.0
    batch_size = 64
    model_dir = '/content/drive/My Drive/DATA7902/Best_model/best_model.pt'
    extra_label = []
    extra_data = []
    reports = []
    max_probs_dict = {}
    test_data_dict = {}
    test_label_dict = {}
    for record in test_records:
        max_probs = []
        test_data = data_dict[record]
        test_label = label_dict[record]
        N_data, S_data, V_data, F_data = DivideClass(test_data, test_label)
        test_x, test_y = Test_Data(N_data, S_data, V_data, F_data)
        test_loader = TestLoader(test_x, test_y, 128, False)
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs, target = data
                # inputs.unsqueeze(0)
                # print(inputs.shape)
                if GPU_device:
                    inputs = inputs.cuda()
                    target = target.cuda()
                    encoder.cuda()
                    classifier.cuda()
                latent = encoder(inputs)
                outputs = classifier(latent)
                loss = classification_loss(outputs, target)
                preds_softmax = F.softmax(outputs, dim=1)
                preds_softmax_np = list(preds_softmax.detach().cpu().numpy())
                max_probs += preds_softmax_np
        max_probs_dict[record] = max_probs
        test_data_dict[record] = test_x
        test_label_dict[record] = test_y
        # print(np.array(max_probs).shape)
    for epoch in range(Epoch):
        if len(extra_data) > 0:
            train_loader = TrainLoader(np.array(extra_data), np.array(extra_label), 50, True)
            count = 0
            encoder.train()
            classifier.train()
            # decoder.train()
            # encoder, decoder, classifier = Pretrain_Causal(encoder, decoder, classifier, train_loader, lr = lr, Epoch = Epoch, alpha = 0.95, beta = 0.05, causal = True)

            for i in range(20):
                for batch_idx, data in enumerate(train_loader):
                    inputs, target = data
                    size = target.size(0)
                    count += size

                    if GPU_device:
                        inputs = inputs.cuda()
                        target = target.cuda()
                        encoder.cuda()
                        classifier.cuda()

                    latent = encoder(inputs)
                    outputs = classifier(latent)
                    loss = classification_loss(outputs, target)
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer1.step()
                    optimizer2.step()
                # print(batch_idx)

        encoder.eval()
        classifier.eval()
        # decoder.eval()
        total = 0
        correct = 0
        sum_loss = 0.0
        all_preds = torch.tensor([])
        targets = torch.tensor([])

        # print(len(test_data))
        # print(len(max_probs))
        # print(len(test_data))
        with torch.no_grad():
            for record in test_records:
                confident_entire = []
                test_x = test_data_dict[record]
                test_y = test_label_dict[record]
                test_loader = TestLoader(test_x, test_y, 128, False)
                for idx, data in enumerate(test_loader):

                    inputs, target = data
                    # inputs.unsqueeze(0)
                    # print(inputs.shape)
                    if GPU_device:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    latent = encoder(inputs)
                    outputs = classifier(latent)
                    loss = classification_loss(outputs, target)
                    preds_softmax = F.softmax(outputs, dim=1)
                    preds_softmax_np = preds_softmax.detach().cpu().numpy()
                    preds_softmax_sort = preds_softmax_np.copy()
                    preds_softmax_sort.sort()
                    if mode == 'prob':
                        confident_entire.append(np.abs(preds_softmax_sort[:, -1] - preds_softmax_sort[:, -2]))
                    elif mode == 'entr':
                        confident_entire.append(-1 * np.sum(preds_softmax_np * np.log(preds_softmax_np + 1e-5), 1))
                    elif mode == 'rand':
                        confident_entire.append(-1 * np.sum(preds_softmax_np * np.log(preds_softmax_np + 1e-5), 1))

                    max_pred_prob, predicted = torch.max(outputs.data, dim=1)
                    # print(max_probs[idx])
                    # print(preds_softmax_np)
                    for j in range(len(inputs)):
                        # print(inputs.shape)
                        # print(j)
                        # print(idx)
                        if max(max_probs_dict[record][idx * 128 + j]) < max(preds_softmax_np[j]):
                            max_probs_dict[record][idx * 128 + j] = preds_softmax_np[j]
                    predicted = np.argmax(max_probs_dict[record][idx * 128:(idx + 1) * 128])
                    # if idx == 0:
                    # all_preds = torch.tensor(max_probs[record][idx*128:(idx+1)*128])
                    # print(all_preds.shape)
                    # else:
                    # print(max_probs[idx])
                    all_preds = torch.cat((all_preds, torch.tensor(max_probs_dict[record][idx * 128:(idx + 1) * 128])),
                                          dim=0)
                    # print(all_preds.shape)
                    targets = torch.cat((targets, target.cpu()), dim=0)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    sum_loss += loss.item()
                confident_entire = np.concatenate(confident_entire, axis=0)
                # print(confident_entire.shape)
                if mode == 'prob':
                    # idx = list(np.argpartition(confident_entire, q_size)[:q_size]) + list(np.argpartition(confident_entire, -1*q_size)[-1*q_size:])
                    # idx = list(np.argpartition(confident_entire, -2*q_size)[-2*q_size:])
                    idx = list(np.argpartition(confident_entire, 2 * q_size)[:2 * q_size])
                    idx = np.array(idx)
                elif mode == 'entr':
                    idx = list(np.argpartition(confident_entire, q_size)[:q_size]) + list(
                        np.argpartition(confident_entire, -1 * q_size)[-1 * q_size:])
                    idx = np.array(idx)
                elif mode == 'rand':
                    idx = np.random.choice(len(confident_entire), q_size * 2, replace=False)
                for i in range(len(test_label_dict[record])):
                    if i in idx:
                        extra_data.append(test_data_dict[record][i])
                        extra_label.append(test_label_dict[record][i])
                test_data_dict[record] = np.delete(test_data_dict[record], idx, axis=0)
                # print(len(test_data))
                test_label_dict[record] = np.delete(test_label_dict[record], idx)
                max_probs_dict[record] = np.delete(max_probs_dict[record], idx, axis=0)
            # print(len(max_probs))
            # print(len(test_data))
            # print(len(test_label))
            print('active learning size: ', len(extra_data))
            acc = 100.0 * correct / total
            print(('Accuracy on test set: %f %% [%d  /  %d]' % (acc, correct, total)))
            cm = confusion_matrix(np.array(targets), all_preds.argmax(dim=1))
            print(cm)
            report = classification_report(np.array(targets), all_preds.argmax(dim=1), digits=3, output_dict=True)
            reports.append(report)
            print(classification_report(np.array(targets), all_preds.argmax(dim=1), digits=3))
        encoder.train()
        classifier.train()
    return reports