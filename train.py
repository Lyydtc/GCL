import os
import sys
import wandb
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.model_selection import KFold

from my_dataset import MyDataset
from my_parser import parsed_args
from my_utils import write_log_file, calculate_metrics, AverageMeter, get_logger
from augment import augment
from BrainUSL import unsupervisedGroupContrast, Model, sameLoss
from model import MyModel, MLP_Decoder, GTCNet


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

        # path
        self.args.pre_model_save_path = os.path.join("point/", self.args.dataset + '_pre-model.pt')
        self.args.model_save_path = os.path.join("point/", self.args.dataset + '_model.pt')

        # dataset
        self.dataset = MyDataset(self.args)
        self.args.in_features = self.dataset.training_graphs.num_features
        self.args.n_max_nodes = self.dataset.n_max_nodes
        self.args.num_classes = self.dataset.training_graphs.num_classes

        # model
        self.model = MyModel(args).to(args.device)
        # use this to help compute loss
        self.BrainUSLModel = Model()
        # downstream
        self.Decoder = MLP_Decoder(self.args).to(self.args.device)

    def pre_train(self):
        self.model.train()
        self.logger = get_logger('logs/' + self.args.dataset + '_pre_train.txt')

        if self.args.load_pre_model:
            print('load model ...')
            self.model.load_state_dict(torch.load(self.args.premodel_save_path))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.pre_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.args.lr_reduce_factor,
                                                               patience=self.args.lr_schedule_patience,
                                                               min_lr=self.args.min_lr,
                                                               verbose=True)

        losses = AverageMeter()
        for epoch in range(1, self.args.pre_epochs + 1):
            start = timer()
            tic = timer()

            batches = self.dataset.create_batches(self.dataset.training_graphs)
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
                scheduler.step(loss)
                print(epoch, index)

            end = timer()
            msg = 'Epoch: {}\t Average loss: {}\t Time: {}'.format(epoch, losses.avg, end-start)
            self.logger.info(msg)

        for k in self.args.__dict__:
            self.logger.info(k + ": " + str(self.args.__dict__[k]))
        torch.save(self.model.state_dict(), self.args.premodel_save_path)

    def train(self):

        self.model.load_state_dict(torch.load(self.args.pre_model_save_path))
        self.logger = get_logger('logs/' + self.args.dataset + '_downstream.txt')

        if self.args.load_GTCmodel:
            print('load GTC ...')
            self.Decoder.load_state_dict(torch.load(self.args.model_save_path))

        Decoder_optimizer = torch.optim.AdamW(self.Decoder.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        Decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Decoder_optimizer, mode='min',
                                                                       factor=self.args.lr_reduce_factor,
                                                                       patience=self.args.lr_schedule_patience,
                                                                       min_lr=self.args.min_lr,
                                                                       verbose=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr * 0.001,
                                      weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=self.args.lr_reduce_factor,
                                                               patience=self.args.lr_schedule_patience,
                                                               min_lr=self.args.min_lr,
                                                               verbose=True)

        min_test_mse = 1e10
        for epoch in range(self.args.epochs):
            if self.args.fine_tuning:
                self.model.train()
            else:
                self.model.eval()

            tic = timer()
            losses = AverageMeter()
            accs = AverageMeter()
            loader = self.dataset.create_batch(self.dataset.training_graphs)
            for index, batch in enumerate(loader):
                Decoder_optimizer.zero_grad()
                # optimizer.zero_grad()
                data = self.dataset.transform_single(batch)
                #
                x1, x2 = self.model.get_embeddings(data, data)
                trues = data['target']
                prediction = self.Decoder(x1, x2)

                loss = F.cross_entropy(prediction, trues, reduction='mean')
                loss.backward()
                Decoder_optimizer.step()
                Decoder_scheduler.step(loss)
                if self.args.fine_tuning:
                    optimizer.step()
                    scheduler.step(loss)
                losses.update(loss.item())

                pred = torch.argmax(prediction.cpu().detach(), dim=1)
                trues = trues.cpu().detach()
                acc = accuracy_score(trues, pred)
                accs.update(acc)

            msg = 'Epoch: {}\t Avg loss: {:.4f}\t Avg accuracy: {:.4f}'.format(epoch, losses.avg, accs.avg)
            self.logger.info(msg)

            if (epoch + 1) % max(1, self.args.iterations // 6) == 0:
                test_mse = self.test()
                if min_test_mse > test_mse and epoch > 20:
                    min_test_mse = test_mse
                    print('save  model...')
                    torch.save(self.Decoder.state_dict(), self.args.model_save_path)

    def kfold_train(self):
        self.model.load_state_dict(torch.load(self.args.premodel_save_path))
        # 假设已经定义了模型 model 和损失函数 criterion，并加载了数据集 data
        # 定义超参数和交叉验证的折数
        num_folds = 10
        num_epochs = self.args.iterations // num_folds
        # patience = num_epochs // 10
        patience = 50
        min_test_loss = 1e8
        # 定义优化器和其他训练相关的参数
        Decoder_optimizer = torch.optim.AdamW(self.Decoder.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        Decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Decoder_optimizer, mode='min',
                                                                       factor=self.args.lr_reduce_factor,
                                                                       patience=self.args.lr_schedule_patience,
                                                                       min_lr=self.args.min_lr,
                                                                       verbose=True)
        # 创建十折交叉验证的分割器
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # 循环进行交叉验证
        data = self.dataset.training_graphs
        for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
            best_val_loss = 1e8
            counter = 0
            # 根据分割得到的训练集和验证集索引，创建对应的数据加载器
            train_data = self.dataset.training_graphs[train_indices.astype(np.int64)]
            val_data = self.dataset.training_graphs[val_indices.astype(np.int64)]
            batches = self.dataset.create_batch(train_data)
            # 训练和验证模型
            for epoch in range(num_epochs):
                loss_sum = 0
                main_index = 0
                max_acc_list = []
                max_auc_list = []
                max_f1_list = []
                # 在训练集上进行训练
                self.Decoder.train()
                if self.args.fine_tuning:
                    self.model.train()
                else:
                    self.model.eval()
                for index, batch_pair in enumerate(batches):
                    Decoder_optimizer.zero_grad()

                    data = self.dataset.transform_single(batch_pair)
                    #
                    if self.args.fine_tuning:
                        x1, x2 = self.model.get_embeddings(data, data)
                    else:
                        with torch.no_grad():
                            x1, x2 = self.model.get_embeddings(data, data)
                    trues = data['target']
                    prediction = self.Decoder(x1, x2)
                    loss = F.cross_entropy(prediction, trues, reduction='mean')
                    loss.backward()
                    Decoder_optimizer.step()
                    loss_sum = loss_sum + loss.item()
                    main_index = main_index + batch_pair.num_graphs
                    pred = torch.argmax(prediction.cpu().detach(), dim=1)
                    trues = trues.cpu().detach()
                    acc = accuracy_score(trues, pred)
                    if self.args.num_classes > 2:
                        aucs = roc_auc_score(trues, prediction.cpu().detach(), multi_class='ovo', average='macro')
                        f1, precision, recall = calculate_metrics(pred, trues, self.args.num_classes)
                    else:
                        aucs = roc_auc_score(trues, pred)
                        f1, precision, recall = calculate_metrics(prediction.cpu().detach(), trues,
                                                                  self.args.num_classes)
                    max_acc_list.append(acc)
                    max_auc_list.append(aucs)
                    max_f1_list.append(f1)
                # 计算平均
                loss = loss_sum / main_index
                acc = np.mean(max_acc_list)
                aucs = np.mean(max_auc_list)
                f1 = np.mean(max_f1_list)
                # 在验证集上评估性能
                val_loss, val_acc = self.validate(val_data)
                # 判断是否提前停止训练

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping!")
                        break
                # 打印训练和验证损失
                # print(f'Fold {fold + 1} | Epoch {epoch + 1}: Train Loss: {loss:.6f} ACC : {acc:.4f} | Val Loss: {val_loss:.6f} Val acc :{val_acc:.4f}')
                write_log_file(self.f,
                               f'Fold {fold + 1} | Epoch {epoch + 1}: Train Loss: {loss:.6f} ACC : {acc:.4f} | Val Loss: {val_loss:.6f} Val acc :{val_acc:.4f}')
            test_loss = self.test()
            if min_test_loss > test_loss:
                min_test_mse = test_loss
                print('save  model...')
                torch.save(self.Decoder.state_dict(), self.args.model_save_path)

    def validate(self, val_data):
        self.model.eval()
        self.Decoder.eval()
        batches = self.dataset.create_batch(val_data)
        main_index = 0
        loss_sum = 0
        acc_list = []
        with torch.no_grad():
            for index, batch_pair in enumerate(batches):
                data = self.dataset.transform_single(batch_pair)
                x1, x2 = self.model.get_embeddings(data, data)
                trues = data['target']
                prediction = self.Decoder(x1, x2)

                loss = F.cross_entropy(prediction, trues, reduction='mean')

                main_index = main_index + batch_pair.num_graphs
                loss_sum = loss_sum + loss.item()

                pred = torch.argmax(prediction.cpu().detach(), dim=1)
                trues = trues.cpu().detach()

                acc = accuracy_score(trues, pred)
                acc_list.append(acc)
            loss = loss_sum / main_index

        return loss, np.mean(acc_list)

    def test(self):
        tic = timer()
        print('\nModel evaluation.')
        self.Decoder.eval()
        self.model.eval()
        batches = self.dataset.create_batch(self.dataset.testing_graphs)
        loss_list = []
        acc_list = []
        auc_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        with torch.no_grad():
            for index, batch_pair in enumerate(batches):
                data = self.dataset.transform_single(batch_pair)
                x1, x2 = self.model.get_embeddings(data, data)
                trues = data['target']
                prediction = self.Decoder(x1, x2)
                loss = F.cross_entropy(prediction, trues, reduction='mean')
                pred = torch.argmax(prediction.cpu().detach(), dim=1).numpy()
                trues = trues.cpu().detach()
                acc = accuracy_score(trues, pred)
                if self.args.num_classes > 2:
                    aucs = roc_auc_score(trues, prediction.cpu().detach(), multi_class='ovo', average='macro')
                    f1, precision, recall = calculate_metrics(pred, trues, self.args.num_classes)
                else:
                    aucs = roc_auc_score(trues, pred)
                    f1, precision, recall = calculate_metrics(prediction.cpu().detach(), trues, self.args.num_classes)

                acc_list.append(acc)
                auc_list.append(aucs)
                f1_list.append(f1)
                precision_list.append(precision)
                recall_list.append(recall)
                loss_list.append(loss.item())

            loss = np.mean(loss_list)
            toc = timer()
            test_results = {
                'test_loss': loss,
                'test_acc': np.mean(acc_list),
                'test_auc': np.mean(auc_list),
                'test_f1': np.mean(f1_list),
                'test_precision': np.mean(precision_list),
                'test_recall': np.mean(recall_list),
                '@time': toc - tic
            }
            if self.args.wandb_activate:
                wandb.log(test_results)
            # write_log_file(self.f,"Test: CEloss = {}\nacc = {} \tauc = {}\tf1={} \tprecision = {}\trecall={} @ {}s\n" \
            #                .format(loss, np.mean(acc_list), np.mean(auc_list), np.mean(f1_list),np.mean(precision_list),np.mean(recall_list),
            #                        toc - tic))
            for k, v in test_results.items():
                write_log_file(self.f, '\t {} = {}'.format(k, v))
            write_log_file(self.f, '\n')
            return loss
