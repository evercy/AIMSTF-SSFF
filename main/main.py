"""
Demo for per-subject experiment
"""
from train import train20
#from decision_level_fusion import decision_fusion
import random
import argparse
import os
import torch
import numpy as np
# deap subject_id: sample number
deap_indices_dict = {1: 2400,
                     2: 2400,
                     3: 2340,
                     4: 2400,
                     5: 2340,
                     6: 2400,
                     7: 2400,
                     8: 2400,
                     9: 2400,
                     10: 2400,
                     11: 2220,
                     12: 2400,
                     13: 2400,
                     14: 2340,
                     15: 2400,
                     16: 2400,
                     17: 2400,
                     18: 2400,
                     19: 2400,
                     20: 2400,
                     21: 2400,
                     22: 2400
                     }

# mahnob subject_id: sample number
mahnob_indices_dict = {1: 1611,
                       2: 1611,
                        3: 1305,
                         4: 1611,
                         5: 1611,
                         6: 1611,
                         7: 1611,
                         8: 1611,
                         9: 1124,
                         10: 1611,
                         11: 1611,
                         13: 1611,
                         14: 1611,
                         16: 1370,
                         17: 1611,
                         18: 1611,
                         19: 1611,
                         20: 1611,
                         21: 1611,
                         22: 1611,
                         23: 1611,
                         24: 1611,
                         25: 1611,
                         27: 1611,
                         28: 1611,
                         29: 1611,
                         30: 1611}
def demo():
    #train
    parser = argparse.ArgumentParser(description='Per-subject experiment')
    parser.add_argument('--dataset', '-d', default='DEAP20', help='The dataset used for evaluation', type=str)
    parser.add_argument('--fusion', default='feature', help='Fusion strategy (feature or decision)', type=str)
    parser.add_argument('--epoch', '-e', default=50, help='The number of epochs in training', type=int)
    parser.add_argument('--batch_size', '-b', default=64, help='The batch size used in training', type=int)
    parser.add_argument('--learn_rate', '-l', default=0.0001, help='Learn rate in training', type=float)
    parser.add_argument('--gpu', '-g', default='True', help='Use gpu or not', type=str)
    parser.add_argument('--file', '-f', default='./results/results.txt', help='File name to save the results', type=str)
    parser.add_argument('--modal', '-m', default='faceeeg', help='Type of data to train', type=str)
    parser.add_argument('--subject', '-s', default=1, help='Subject id', type=int)
    parser.add_argument('--label', default='4class', help='valence or arousal', type=str)
    parser.add_argument('--random-seed', type=int, default=2023)
    #dataset      self.label == '4class':
    parser.add_argument('--withoutbaseline', '-wb', default=False, help='delete base_signals or not', type=bool)
    parser.add_argument('--alpha_weight', '-alpha', default=0, help='diff_weight_Loss', type=float)
    parser.add_argument('--beta_weight', '-beta', default=0, help='sim_weight_Loss', type=float)
    parser.add_argument('--gamma_weight', '-gamma', default=0, help='diff_weight_Loss', type=float)
    #model
    parser.add_argument('--face_feature_size', default=128, help='Face feature size', type=int)
    parser.add_argument('--eeg_feature_size', default=128, help='EEG feature size', type=int)
    parser.add_argument('--fin_feature_size', default=128, help='Finally feature size', type=int)

    parser.add_argument('--pretrain', default='True', help='Use pretrained CNN', type=str)
    parser.add_argument('--sampling_rate', default=128, help='sampling_rate', type=int)
    parser.add_argument('--eeg_channel', default=32, help='expend channel', type=int)
    parser.add_argument('--t_channel', default=128, help='data_length', type=int)
    parser.add_argument('--s_kernel', default=3, help='channel_kernel', type=int)
    parser.add_argument('--t_kernel', default=3, help='time_kernel', type=int)
    parser.add_argument('--s_channel', default=32, help='EEG_channel', type=int)
    parser.add_argument('--hidden_dim', default=24, help='the channel of the gated', type=int)
    parser.add_argument('--kernel_size', default=3, help='gated conv', type=int)
    parser.add_argument('--num_layers', default=6, help='num_layers', type=int)
    parser.add_argument('--fc_hidden', default=48, help='fc_hidden', type=int)

    #fusion
    parser.add_argument('--dropout', default=0.5, help='dropout', type=float)
    #eeg
    parser.add_argument('--dropout_rate', default=0.1, help='dropout_rate', type=float)
    parser.add_argument('--num_classes', default=4, help='num_layers', type=int)

    args = parser.parse_args()
    use_gpu = True if args.gpu == 'True' else False
    pretrain = True if args.pretrain == 'True' else False
    avg_val_acc = []
    if args.dataset == 'DEAP20':
        for subject in deap_indices_dict:
            print(subject)
            if args.dataset == 'DEAP20':
                indices = list(range(deap_indices_dict[subject]))
            if not os.path.exists(f'./results/'):
                os.mkdir(f'./results/')
            if not os.path.exists(f'./results/{args.dataset}/'):
                os.mkdir(f'./results/{args.dataset}/')
            if not os.path.exists(f'./results/{args.dataset}/{args.modal}/'):
                os.mkdir(f'./results/{args.dataset}/{args.modal}/')
            avg_val_acc_allsubject = []
            print(
                f'{args.label}_s{subject}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}'
                f'_alpha_weight{args.alpha_weight}beta_weight{args.beta_weight}gamma_weight{args.gamma_weight}'
                f'_dropout{args.dropout}_num_classes{args.num_classes}/'
                f'{args.dataset}_{args.modal}')
            if not os.path.exists(
                    f'./results/{args.dataset}/{args.modal}/{args.label}_s{subject}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}_alpha_weight{args.alpha_weight}_dropout{args.dropout}/'):
                os.mkdir(
                    f'./results/{args.dataset}/{args.modal}/{args.label}_s{subject}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}_alpha_weight{args.alpha_weight}_dropout{args.dropout}/')

            best_accuracy = train20(modal=args.modal, dataset=args.dataset, subject=subject, l=args.label, epoch_num=args.epoch,
                                  lr=args.learn_rate, batch_size=args.batch_size,
                                  file_name=f'./results/{args.dataset}/{args.modal}/'
                                            f'{args.label}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}_alpha_weight{args.alpha_weight}_dropout{args.dropout}/'
                                            f'{args.dataset}_{args.modal}_{args.label}_s{args.subject}',
                                             indices=indices, face_feature_size=args.face_feature_size,
                                             use_gpu=use_gpu, pretrain=pretrain,
                                  withoutbaseline=args.withoutbaseline,
                                 sampling_rate=args.sampling_rate,eeg_channel=args.eeg_channel,s_channel=args.s_channel,
                                 s_kernel=args.s_kernel, t_kernel=args.t_kernel,t_channel = args.t_channel,
                                 hidden_dim=args.hidden_dim, kernel_size=args.kernel_size,
                                 num_layers=args.num_layers, eeg_feature_size=args.eeg_feature_size,
                                 fin_feature_size = args.fin_feature_size,dropout=args.dropout,num_classes=args.num_classes,
                                 alpha_weight=args.alpha_weight, beta_weight=args.beta_weight,gamma_weight=args.gamma_weight)
            avg_val_acc.append(best_accuracy)
            avg_val_acc_current_subject = np.mean(avg_val_acc)
            print("avg_val_acc_subject: {:.4f}".format(avg_val_acc_current_subject))
            avg_val_acc_subject = np.mean(avg_val_acc)
            #avg_val_acc_allsubject.append(avg_val_acc_subject)
            print("avg_val_acc_subject: {:.4f}".format(avg_val_acc_subject))
    else:
        for subject in mahnob_indices_dict:
            print(subject)
            if args.dataset == 'MAHNOB20':
                indices = list(range(mahnob_indices_dict[subject]))
            if args.dataset != 'DEAP20' and args.dataset != 'MAHNOB20':
                random.shuffle(indices)
            if not os.path.exists(f'./results/'):
                os.mkdir(f'./results/')
            if not os.path.exists(f'./results/{args.dataset}/'):
                os.mkdir(f'./results/{args.dataset}/')
            if not os.path.exists(f'./results/{args.dataset}/{args.modal}/'):
                os.mkdir(f'./results/{args.dataset}/{args.modal}/')
            avg_val_acc_allsubject = []

            print(
                f'{args.label}_s{subject}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}'
                f'_alpha_weight{args.alpha_weight}beta_weight{args.beta_weight}gamma_weight{args.gamma_weight}'
                f'_dropout{args.dropout}_num_classes{args.num_classes}/'
                f'{args.dataset}_{args.modal}')
            if not os.path.exists(
                    f'./results/{args.dataset}/{args.modal}/{args.label}_s{subject}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}_alpha_weight{args.alpha_weight}_dropout{args.dropout}/'):
                os.mkdir(
                    f'./results/{args.dataset}/{args.modal}/{args.label}_s{subject}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}_alpha_weight{args.alpha_weight}_dropout{args.dropout}/')
            best_accuracy = train20(modal=args.modal, dataset=args.dataset, subject=subject, l=args.label,
                                    epoch_num=args.epoch,
                                    lr=args.learn_rate, batch_size=args.batch_size,
                                    file_name=f'./results/{args.dataset}/{args.modal}/'
                                              f'{args.label}_bs_{args.batch_size}_lr_{args.learn_rate}_num_layers_{args.num_layers}_alpha_weight{args.alpha_weight}_dropout{args.dropout}/'
                                              f'{args.dataset}_{args.modal}_{args.label}_s{args.subject}',
                                    indices=indices, face_feature_size=args.face_feature_size,
                                    use_gpu=use_gpu, pretrain=pretrain,
                                    withoutbaseline=args.withoutbaseline,
                                    sampling_rate=args.sampling_rate, eeg_channel=args.eeg_channel,
                                    s_channel=args.s_channel,
                                    s_kernel=args.s_kernel, t_kernel=args.t_kernel, t_channel=args.t_channel,
                                    hidden_dim=args.hidden_dim, kernel_size=args.kernel_size,
                                    num_layers=args.num_layers, eeg_feature_size=args.eeg_feature_size,
                                    fin_feature_size=args.fin_feature_size, dropout=args.dropout,num_classes=args.num_classes,
                                    alpha_weight=args.alpha_weight, beta_weight=args.beta_weight,gamma_weight=args.gamma_weight)
            avg_val_acc.append(best_accuracy)
            avg_val_acc_current_subject = np.mean(avg_val_acc)
            print("avg_val_acc_subject: {:.4f}".format(avg_val_acc_current_subject))
        avg_val_acc_subject = np.mean(avg_val_acc)
        # avg_val_acc_allsubject.append(avg_val_acc_subject)
        print("avg_val_acc_subject: {:.4f}".format(avg_val_acc_subject))

if __name__ == '__main__':
    demo()