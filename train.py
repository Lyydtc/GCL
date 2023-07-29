import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import random
from tqdm import tqdm
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from my_dataset import MyDataset
from my_utils import calculate_metrics, AverageMeter, get_logger, sample_mask
from augment import augment
from BrainUSL import unsupervisedGroupContrast, Model, sameLoss
from model import MyModel, MLP_Decoder, MLP_Decoder5
from info_nce import info_nce


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    # random.seed(seed) # augmentation need random


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        setup_seed(self.args.seed)
        self.timestamp = time.strftime('%m_%d_%H_%M_%S', time.localtime())

        # path
        self.pre_model_save_path = os.path.join("point/", self.args.dataset +
                                                '_pre_model_' + self.timestamp + '.pt')
        self.decoder_save_path = os.path.join("point/", self.args.dataset + '_decoder_' + self.timestamp + '.pt')

        # dataset
        self.dataset = MyDataset(self.args)
        self.args.in_features = self.dataset.train_graphs.num_features
        if self.args.task == 'cls':
            self.args.num_classes = self.dataset.train_graphs.num_classes

        # model
        self.model = MyModel(args).to(args.device)
        # use this to help compute loss
        self.BrainUSLModel = Model()

    def pre_train(self, pre_model_path):
        self.model.train()
        self.logger = get_logger('logs/' + self.args.dataset + '_pre_train_' + self.timestamp + '.txt')
        for k in self.args.__dict__:
            self.logger.info(k + ": " + str(self.args.__dict__[k]))

        if self.args.load_pre_model:
            print('Loading pre-model ...')
            self.model.load_state_dict(torch.load(pre_model_path))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.pre_lr)
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.args.pre_lr,
        #     momentum=0.9,
        #     weight_decay=0.0001,
        #     nesterov=False
        # )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        factor=0.1,
        #                                                        patience=200,
        #                                                        verbose=True)
        min_loss = 10
        for epoch in range(1, self.args.pre_epochs + 1):
            losses = AverageMeter()
            start = timer()
            tic = timer()

            batches = tqdm(self.dataset.create_batches(self.dataset.train_graphs))
            for index, data in enumerate(batches):
                optimizer.zero_grad()

                # data[0] and data[1] are two batches
                # augmentation and preparation
                [G1, G2] = data
                G1_aug1 = augment(G1, self.args)
                G1_aug2 = augment(G1, self.args)
                G2_aug1 = augment(G2, self.args)
                G2_aug2 = augment(G2, self.args)

                if self.args.switch:
                    G1_aug1 = self.dataset.transform_single_for_switch(G1_aug1)
                    G1_aug2 = self.dataset.transform_single_for_switch(G1_aug2)
                    G2_aug1 = self.dataset.transform_single_for_switch(G2_aug1)
                    G2_aug2 = self.dataset.transform_single_for_switch(G2_aug2)
                else:
                    G1_aug1 = self.dataset.transform_single(G1_aug1)
                    G1_aug2 = self.dataset.transform_single(G1_aug2)
                    G2_aug1 = self.dataset.transform_single(G2_aug1)
                    G2_aug2 = self.dataset.transform_single(G2_aug2)
                # print('data prepared! time:{:.2f}'.format(timer() - tic))
                # tic = timer()

                # compute output
                y_pred_11, y_pred_21 = self.model(G1_aug1, G2_aug1)
                y_pred_12, y_pred_22 = self.model(G1_aug2, G2_aug2)
                # print('model done! time:{:.2f}'.format(timer() - tic))
                # tic = timer()

                # contrastive loss
                # infoNCE loss
                # same_Loss1 = info_nce(y_pred_11, y_pred_12)
                # same_Loss2 = info_nce(y_pred_21, y_pred_22)
                # 图核loss
                same_Loss1 = sameLoss(y_pred_11, y_pred_12)
                same_Loss2 = sameLoss(y_pred_21, y_pred_22)
                # 利用先验相似度增加正样本数量 loss
                x1 = G1_aug1['dense_x']
                x1 = torch.matmul(x1, x1.transpose(1, 2))
                sim_matrix1 = self.BrainUSLModel.get_label_matrix_from_sim(G1_aug1['adj'], y=self.args.y)
                sim_matrix2 = self.BrainUSLModel.get_label_matrix_from_sim(x1, y=self.args.y)
                sim_matrix_1 = sim_matrix1 & sim_matrix2
                x2 = G2_aug1['dense_x']
                x2 = torch.matmul(x2, x2.transpose(1, 2))
                sim_matrix1 = self.BrainUSLModel.get_label_matrix_from_sim(G2_aug1['adj'], y=self.args.y)
                sim_matrix2 = self.BrainUSLModel.get_label_matrix_from_sim(x2, y=self.args.y)
                sim_matrix_2 = sim_matrix1 & sim_matrix2
                ugc_loss1 = unsupervisedGroupContrast(y_pred_11, y_pred_12, sim_matrix_1, self.args.T)
                ugc_loss2 = unsupervisedGroupContrast(y_pred_21, y_pred_22, sim_matrix_2, self.args.T)
                # total loss
                loss = 1.0 * (same_Loss1 + same_Loss2) / 2 + 1 * (ugc_loss1 + ugc_loss2) / 2
                losses.update(loss.item())
                # print('loss done! time:{:.2f}'.format(timer() - tic))
                # tic = timer()

                loss.backward()
                optimizer.step()
                # scheduler.step(loss)
                batches.set_description('Epoch: {}, Iter: {}'.format(epoch, index))
                batches.set_postfix(loss=float(loss))

            end = timer()
            msg = 'Epoch: {}\t Average loss: {:.4f}\t Time: {:.4f}'.format(epoch, losses.avg, end - start)
            self.logger.info(msg)
            if losses.avg < min_loss:
                min_loss = losses.avg
                torch.save(self.model.state_dict(), self.pre_model_save_path)
                self.logger.info('Model saved!')

    def kfold_train(self, pre_model_path):
        self.logger = get_logger('logs/' + self.args.dataset + '_downstream_' + self.timestamp + '.txt')
        self.logger.info(pre_model_path)
        for k in self.args.__dict__:
            self.logger.info(k + ": " + str(self.args.__dict__[k]))

        data = self.dataset.train_graphs
        if self.args.task == 'cls':
            test_data = self.dataset.test_graphs
        elif self.args.task == 'gsl':
            test_data = self.dataset.test_graphs

        best_acc = 0
        accs = AverageMeter()  # 统计每个fold的最佳
        best_mse = 100
        mses = AverageMeter()  # 统计每个fold的最佳

        self.logger.info('{} fold training...'.format(self.args.num_folds))
        kfold = KFold(n_splits=self.args.num_folds, shuffle=True)

        for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
            fold = fold + 1

            # data and loader
            train_data = data[train_indices.astype(np.int64)]
            val_data = data[val_indices.astype(np.int64)]

            # train and get indicators
            if self.args.task == 'cls':
                train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)

                # this decoder and acc is already the best in this fold
                decoder, acc = self.ds_train(self.logger, pre_model_path, fold, train_loader, val_data)
                accs.update(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_decoder = decoder
                    self.logger.info('New decoder selected on fold {}'.format(fold))

            elif self.args.task == 'gsl':
                train_loader = self.dataset.create_batches(train_data)
                decoder, mse = self.ds_train(self.logger, pre_model_path, fold, train_loader, val_data)
                mses.update(mse)
                if mse < best_mse:
                    best_mse = mse
                    best_decoder = decoder
                    self.logger.info('New decoder selected on fold {}'.format(fold))

        if self.args.task == 'cls':
            self.logger.info('Training finished! Tht best accuracy on val is {:.4f}, '
                             'the average acc is {:.4f}'.format(best_acc, accs.avg))
        elif self.args.task == 'gsl':
            self.logger.info('Training finished! Tht best mse on val is {:.4f}, '
                             'the average mse is {:.4f}'.format(best_mse, mses.avg))

        # the best model on val for test
        if self.args.task == 'cls':
            test_acc, test_loss = self.ds_test(best_decoder, test_data)
            self.logger.info('The accuracy on test data is {:.6f}'.format(test_acc))
        elif self.args.task == 'gsl':
            _, test_mse = self.ds_test(best_decoder, test_data)
            self.logger.info('The mse on test data is {:.6f}'.format(test_mse))

        torch.save(best_decoder.state_dict(), self.decoder_save_path)
        self.logger.info('Decoder saved!')

    def ds_train(self, logger, pre_model_path, fold, train_loader, test_data):
        """downstream train classification"""
        if self.args.decoder == 'mlp4':
            self.decoder = MLP_Decoder(self.args).to(self.args.device)
        elif self.args.decoder == 'mlp5':
            self.decoder = MLP_Decoder5(self.args).to(self.args.device)

        self.model.load_state_dict(torch.load(pre_model_path))
        self.logger = logger

        Decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        Decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Decoder_optimizer, mode='min',
                                                                       factor=self.args.lr_reduce_factor,
                                                                       patience=self.args.lr_schedule_patience,
                                                                       min_lr=self.args.min_lr,
                                                                       verbose=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr * 0.1,
                                      weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.args.lr_reduce_factor,
                                                               patience=self.args.lr_schedule_patience,
                                                               min_lr=self.args.min_lr,
                                                               verbose=True)

        best_test_acc = 0
        best_test_mse = 100
        best_decoder = self.decoder
        for epoch in range(1, self.args.epochs + 1):
            if self.args.fine_tuning:
                self.model.train()
            else:
                self.model.eval()

            tic = timer()
            losses = AverageMeter()
            accs = AverageMeter()
            train_loader = tqdm(train_loader)
            for index, batch in enumerate(train_loader):

                Decoder_optimizer.zero_grad()
                if self.args.fine_tuning:
                    optimizer.zero_grad()

                if self.args.task == 'cls':
                    g = self.dataset.transform_single(batch)
                    trues = g['target']
                    input1, input2 = self.model.get_embeddings(g, g)
                    output = self.decoder(input1, input2)
                    loss = F.cross_entropy(output, trues, reduction='mean')
                elif self.args.task == 'gsl':
                    [g1, g2] = batch
                    trues = self.dataset.train_graphs.norm_ged[g1.i, g2.i].to(self.args.device)
                    g1 = self.dataset.transform_single(g1)
                    g2 = self.dataset.transform_single(g2)
                    input1, input2 = self.model.get_embeddings(g1, g2)
                    output = self.decoder(input1, input2)
                    loss = F.mse_loss(output.squeeze(), trues)

                loss.backward()
                losses.update(loss)

                Decoder_optimizer.step()
                Decoder_scheduler.step(loss)
                if self.args.fine_tuning:
                    optimizer.step()
                    scheduler.step(loss)

                if self.args.task == 'cls':
                    pred = torch.argmax(output.cpu().detach(), dim=1)
                    trues = trues.cpu().detach()
                    acc = accuracy_score(trues, pred)
                    accs.update(acc)
                    train_loader.set_postfix(loss=float(loss), acc=float(acc))
                elif self.args.task == 'gsl':
                    train_loader.set_postfix(mse_loss=float(loss))

            if self.args.task == 'cls':
                msg = 'Fold: {}\t Epoch: {}\t Avg loss: {:.4f}\t Avg accuracy: {:.4f}\t ' \
                      'Time: {:.4f}'.format(fold, epoch, losses.avg, accs.avg, timer() - tic)
            elif self.args.task == 'gsl':
                msg = 'Fold: {}\t Epoch: {}\t Avg loss: {:.4f}\t ' \
                      'Time: {:.4f}'.format(fold, epoch, losses.avg, timer() - tic)
            self.logger.info(msg)

            if epoch % 2 == 0:
                if self.args.task == 'cls':
                    test_acc, test_loss = self.ds_test(self.decoder, test_data)
                    self.logger.info('Test acc: {:.4f}\t loss: {:.4f}'.format(test_acc, test_loss))
                    if best_test_acc < test_acc and epoch > 5:
                        best_test_acc = test_acc
                        best_decoder = self.decoder
                        self.logger.info('New best decoder within a fold!')
                elif self.args.task == 'gsl':
                    _, test_mse = self.ds_test(self.decoder, test_data)
                    self.logger.info('Test mse: {:.4f}'.format(test_mse))
                    if best_test_mse > test_mse and epoch > 5:
                        best_test_mse = test_mse
                        best_decoder = self.decoder
                        self.logger.info('New best decoder within a fold!')

        if self.args.task == 'cls':
            return best_decoder, best_test_acc
        elif self.args.task == 'gsl':
            return best_decoder, best_test_mse

    def ds_test(self, decoder, test_data):
        decoder.eval()
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size)

        accs = AverageMeter()
        losses = AverageMeter()
        p_bar = tqdm(test_loader, colour='green')
        for g in p_bar:
            if self.args.task == 'cls':
                g = self.dataset.transform_single(g)
                input1, input2 = self.model.get_embeddings(g, g)
                output = decoder(input1, input2)
                trues = g['target']
                loss = F.cross_entropy(output, trues, reduction='mean')
                pred = torch.argmax(output.cpu().detach(), dim=1).numpy()
                trues = trues.cpu().detach()
                acc = accuracy_score(trues, pred)

                p_bar.set_postfix(acc=acc)

                # if self.args.num_classes > 2:
                #     aucs = roc_auc_score(trues, prediction.cpu().detach(), multi_class='ovo', average='macro')
                #     f1, precision, recall = calculate_metrics(pred, trues, self.args.num_classes)
                # else:
                #     aucs = roc_auc_score(trues, pred)
                #     f1, precision, recall = calculate_metrics(prediction.cpu().detach(), trues, self.args.num_classes)

                accs.update(acc)
                losses.update(loss)
            elif self.args.task == 'gsl':
                g2 = g

                # get g1 (batched) from train_graphs
                # because norm_ged only support train-train or train-test
                dataset_size = len(self.dataset.train_graphs)
                # shuffle all the idx
                shuffled_idx = shuffle(np.array(range(dataset_size)), random_state=0)
                # slice the top batch_size idx
                train_idx = shuffled_idx[:int(g.num_graphs)].tolist()
                # get the dataset
                g1 = self.dataset.train_graphs[sample_mask(train_idx, dataset_size)]
                # create batch
                g1 = Batch.from_data_list([g for g in g1])

                trues = self.dataset.train_graphs.norm_ged[g1.i, g2.i].to(self.args.device)

                input1 = self.dataset.transform_single(g1)
                input2 = self.dataset.transform_single(g2)
                x1, x2 = self.model.get_embeddings(input1, input2)
                output = decoder(x1, x2)
                loss = F.mse_loss(output.squeeze(), trues)

                p_bar.set_postfix(mse=float(loss))
                losses.update(loss)

        return accs.avg, losses.avg
