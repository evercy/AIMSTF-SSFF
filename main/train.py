"""
Train and test
"""
from torchnet import meter
import torch
import os
from torch.utils.data import DataLoader
import time

from dataset import MAHNOB20, DEAP20
from utils import *
from EFFusion import EFFusionNet
from functional import DiffLoss

import torch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # Use Xavier initialization for linear layers
        torch.nn.init.zeros_(m.bias)  # Initialize biases with zeros

def f_norm_loss(X, Y):
    return torch.norm(X - Y, 'fro')

# Define the subspace orthogonality loss
def subspace_orthogonality_loss(orthogonal_matrix):
    identity = torch.eye(orthogonal_matrix.size(1))
    return torch.norm(orthogonal_matrix - torch.mm(orthogonal_matrix, orthogonal_matrix.t()) - identity, 'fro')

def central_moments(data, order):

    mean = torch.mean(data)
    centered_data = data - mean
    moments = torch.pow(centered_data, order)
    central_moment = torch.mean(moments)
    return central_moment

def central_moment_discrepancy(X, Y, orders):
    discrepancy = 0
    for order in orders:
        cm_X = central_moments(X, order)
        cm_Y = central_moments(Y, order)
        discrepancy += torch.abs(cm_X - cm_Y)
    return discrepancy






def train20(modal, dataset, subject, l, epoch_num, lr, batch_size, file_name, indices,face_feature_size,
          use_gpu, pretrain,
          withoutbaseline,
          sampling_rate,eeg_channel,s_channel,
          s_kernel, t_kernel,t_channel,
          hidden_dim, kernel_size,
          num_layers, eeg_feature_size,fin_feature_size, dropout,num_classes,
          alpha_weight, beta_weight,gamma_weight):

    if use_gpu:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    directory = file_name.split('/')[-2]

    if not os.path.exists(f'./results/{dataset}/{modal}/'+directory):
        os.mkdir(f'./results/{dataset}/{modal}/'+directory)

    if dataset == 'DEAP20':
        ############## per-subjects ##############
        #mean_std_transform = MeanStdNormalize()
        train_data = DEAP20(modal=modal,subject=subject,kind='train',indices=indices, label=l)
        val_data = DEAP20(modal=modal,subject=subject,kind='val',indices=indices, label=l)
    if dataset == 'MAHNOB20':
        ############## per-subject #################
        train_data = MAHNOB20(modal=modal, subject=subject, kind='train', indices=indices, label=l)
        val_data = MAHNOB20(modal=modal, subject=subject,  kind='val', indices=indices, label=l)

    model = EFFusionNet(face_feature_size=face_feature_size, pretrain=pretrain,
                        sampling_rate=sampling_rate,eeg_channel=eeg_channel, s_channel=s_channel,
                        s_kernel = s_kernel, t_kernel = t_kernel, t_channel = t_channel,
                        hidden_dim=hidden_dim, kernel_size=kernel_size, bias=True, dtype=torch.cuda,
                        num_layers=num_layers, eeg_feature_size=eeg_feature_size, fin_feature_size=fin_feature_size,
                        dropout=dropout,num_classes=num_classes).to(device)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8)
    # criterion and optimizer
    if num_classes == 1:
        criterion_loss = torch.nn.BCELoss()
    else:
        criterion_loss = torch.nn.CrossEntropyLoss()
    orders = [1, 2]  # Compute for 1st and 2nd order moments
    diff_loss = DiffLoss()
    lr = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # meters
    loss_meter = meter.AverageValueMeter()
    best_accuracy = 0
    best_epoch = 0

    # train
    if num_classes == 1:
        for epoch in range(epoch_num):
            model.train()
            #since = time.time()
            pred_label = []
            true_label = []

            loss_meter.reset()
            for ii, (data, label) in enumerate(train_loader):
                # train model
                if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                    input = (data[0].float().to(device), data[1].float().to(device))
                else:
                    input = data.float().to(device)
                label = label.float().to(device)
                eeg_output, face_output, eeg_com_output, face_com_output, _, all_feature_fc_2,pred = model(input)
                #pred = pred.float()
                # dim(pred.shape,label.shape)
                pred = pred.float()
                loss_pred = criterion_loss(pred, label).to(device)
                loss_diff_eeg_pc = diff_loss(eeg_output, eeg_com_output).to(device)
                loss_diff_face_pc = diff_loss(face_output, face_com_output).to(device)
                loss_diff_ef = diff_loss(eeg_output, face_output).to(device)

                loss_sim = central_moment_discrepancy(eeg_com_output, face_com_output, orders).to(device)

                loss = loss_pred + alpha_weight * (
                            loss_diff_eeg_pc + loss_diff_face_pc) + gamma_weight * loss_diff_ef + beta_weight * loss_sim
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # meters update
                loss_meter.add(loss.item())
                pred = (pred >= 0.5).float().to(device).data
                # pred = (pred >= 0.5).float().to(device).data
                pred_label.append(pred)
                true_label.append(label)

            pred_label = torch.cat(pred_label,0)
            true_label = torch.cat(true_label,0)

            train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
            #val
            model.eval()

            val_loss_meter = meter.AverageValueMeter()
            pred_label1 = []
            true_label1 = []
            val_loss_meter.reset()
            with torch.no_grad():
                for iii, (data1, label1) in enumerate(val_loader):
                    if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                        input1 = (data1[0].float().to(device), data1[1].float().to(device))
                    else:
                        input1 = data1.float().to(device)

                    label1 = label1.float().to(device)
                    eeg_output1, face_output1, eeg_com_output1, face_com_output1, _, all_feature_fc_2,pred1 = model(input1)
                    #pred1 = pred1.float()
                    # dim(pred.shape,label.shape)
                    pred1 = pred1.float()
                    loss_pred = criterion_loss(pred1, label1).to(device)

                    loss_diff_eeg_pc = diff_loss(eeg_output1, eeg_com_output1).to(device)
                    loss_diff_face_pc = diff_loss(face_output1, face_com_output1).to(device)
                    loss_diff_ef = diff_loss(eeg_output1, face_output1).to(device)

                    loss_sim = central_moment_discrepancy(eeg_com_output1, face_com_output1, orders).to(device)
                    val_loss = loss_pred + loss_pred + alpha_weight * (
                            loss_diff_eeg_pc + loss_diff_face_pc) + gamma_weight * loss_diff_ef + beta_weight * loss_sim

                    pred1 = (pred1 >= 0.5).float().to(device).data
                    pred_label1.append(pred1)
                    true_label1.append(label1)
                    val_loss_meter.add(val_loss.item())

                pred_label1 = torch.cat(pred_label1, 0)
                true_label1 = torch.cat(true_label1, 0)
                val_accuracy = torch.sum(pred_label1 == true_label1).type(torch.FloatTensor) / true_label1.size(0)
                print(
                    'Epoch: train{} | train accuracy: {:.4f} | train loss: {:.2f} | val accuracy: {:.4f} | val loss: {:.2f}'.format(
                        epoch, train_accuracy.item(), loss_meter.value()[0], val_accuracy.item(),
                        val_loss_meter.value()[0]))

            #time_elapsed = time.time() - since
            #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_epoch = epoch
                    model.save(f"{file_name}_best.pth")
        model.save(f'{file_name}.pth')
    else:
        for epoch in range(epoch_num):
            model.train()
            # since = time.time()
            pred_label = []
            true_label = []

            loss_meter.reset()
            for ii, (data, label) in enumerate(train_loader):
                # train model
                if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                    input = (data[0].float().to(device), data[1].float().to(device))
                else:
                    input = data.float().to(device)
                label = label.float().to(device)
                eeg_output, face_output, eeg_com_output, face_com_output, _, all_feature_fc_2, pred = model(input)
                # pred = pred.float()
                # dim(pred.shape,label.shape)


                loss_pred = criterion_loss(pred, label.long()).to(device)
                loss_diff_eeg_pc = diff_loss(eeg_output, eeg_com_output).to(device)
                loss_diff_face_pc = diff_loss(face_output, face_com_output).to(device)
                loss_diff_ef = diff_loss(eeg_output, face_output).to(device)

                loss_sim = central_moment_discrepancy(eeg_com_output, face_com_output, orders).to(device)

                loss = loss_pred + alpha_weight * (
                            loss_diff_eeg_pc + loss_diff_face_pc) + gamma_weight * loss_diff_ef + beta_weight * loss_sim
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # meters update
                loss_meter.add(loss.item())
                _, pred = torch.max(pred, 1)
                # pred = (pred >= 0.5).float().to(device).data
                pred_label.append(pred)
                true_label.append(label)
            pred_label = torch.cat(pred_label, 0)
            true_label = torch.cat(true_label, 0)

            train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)

            # val
            model.eval()

            val_loss_meter = meter.AverageValueMeter()
            pred_label1 = []
            true_label1 = []
            val_loss_meter.reset()
            with torch.no_grad():
                for iii, (data1, label1) in enumerate(val_loader):
                    if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                        input1 = (data1[0].float().to(device), data1[1].float().to(device))
                    else:
                        input1 = data1.float().to(device)

                    label1 = label1.float().to(device)
                    eeg_output1, face_output1, eeg_com_output1, face_com_output1, _, all_feature_fc_2, pred1 = model(
                        input1)
                    # pred1 = pred1.float()
                    # dim(pred.shape,label.shape)
                    loss_pred = criterion_loss(pred1, label1.long()).to(device)

                    loss_diff_eeg_pc = diff_loss(eeg_output1, eeg_com_output1).to(device)
                    loss_diff_face_pc = diff_loss(face_output1, face_com_output1).to(device)
                    loss_diff_ef = diff_loss(eeg_output1, face_output1).to(device)

                    loss_sim = central_moment_discrepancy(eeg_com_output1, face_com_output1, orders).to(device)
                    # loss_sim = sim_loss(eeg_com_output.to(torch.long),face_com_output.to(torch.long)).to(device)
                    # val_loss = loss_pred + alpha_weight * loss_diff_ef + beta_weight * loss_sim
                    val_loss = loss_pred + loss_pred + alpha_weight * (
                            loss_diff_eeg_pc + loss_diff_face_pc) + gamma_weight * loss_diff_ef + beta_weight * loss_sim
                    _, pred1 = torch.max(pred1, 1)
                    # pred1 = (pred1 >= 0.5).float().to(device).data
                    pred_label1.append(pred1)
                    true_label1.append(label1)
                    val_loss_meter.add(val_loss.item())

                pred_label1 = torch.cat(pred_label1, 0)
                true_label1 = torch.cat(true_label1, 0)
                val_accuracy = torch.sum(pred_label1 == true_label1).type(torch.FloatTensor) / true_label1.size(0)

                print(
                    'Epoch: train{} | train accuracy: {:.4f} | train loss: {:.2f} | val accuracy: {:.4f} | val loss: {:.2f}'.format(
                        epoch, train_accuracy.item(), loss_meter.value()[0], val_accuracy.item(),
                        val_loss_meter.value()[0]))
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_epoch = epoch
                    model.save(f"{file_name}_best.pth")
        model.save(f'{file_name}.pth')
    print('best_epoch={} best_accuracy={:.4f}'.format(best_epoch, best_accuracy))


    return best_accuracy