import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from math import sqrt
from face_network import FaceFeatureExtractor
from bio_network import DualSpatialTimeModel

class EEGPrivate(nn.Module):
    def __init__(self, eeg_feature_size, dropout):
        super(EEGPrivate, self).__init__()
        self.private_eeg = nn.Sequential(
            nn.Linear(eeg_feature_size, eeg_feature_size//2),
            #nn.Dropout(dropout),
            nn.Linear(eeg_feature_size//2, eeg_feature_size // 4),
            #nn.Dropout(dropout),
            #nn.Linear(eeg_feature_size//4, eeg_feature_size // 2),
        )

    def forward(self, x):
        #print(x.shape)
        out_eeg = self.private_eeg(x)
        return out_eeg


class FacePrivate(nn.Module):
    def __init__(self, face_feature_size, dropout):
        super(FacePrivate, self).__init__()
        self.private_face = nn.Sequential(
            nn.Linear(face_feature_size, face_feature_size//2),
            #nn.Dropout(dropout),
            nn.Linear(face_feature_size//2, face_feature_size // 4),
            #nn.Dropout(dropout),
            #nn.Linear(face_feature_size // 4, face_feature_size // 2),
        )

    def forward(self, x):
        #print(x.shape)
        out_eeg = self.private_face(x)
        return out_eeg


class CommonNet(nn.Module):
    def __init__(self, eeg_feature_size, dropout):
        super(CommonNet, self).__init__()
        self.common_net = nn.Sequential(
            nn.Linear(eeg_feature_size, eeg_feature_size//2),
            #nn.Dropout(dropout),
            nn.Linear(eeg_feature_size//2, eeg_feature_size // 4),
            #nn.Dropout(dropout),
            #nn.Linear(eeg_feature_size // 4, eeg_feature_size // 2),
        )

    def forward(self, x):

        out_eeg = self.common_net(x)

        return out_eeg


class ScoreNet(nn.Module):
    def __init__(self,fin_feature_size):
        super(ScoreNet, self).__init__()
        self.score_generate = nn.Sequential(
            nn.Linear(fin_feature_size//4*3, fin_feature_size//4),
        )
    def forward(self, x):
        #print('x',x.shape)
        x_score = self.score_generate(x)
        #attention_weights = torch.softmax(x_score, dim=1)

        return x_score



class EFFusionNet(nn.Module):
    def __init__(self, face_feature_size=128, pretrain=True,
                 sampling_rate=128,eeg_channel=32, s_channel=32,
                 s_kernel=3, t_kernel=3, t_channel=128,
                  hidden_dim=24, kernel_size=3, bias=True, dtype=torch.cuda,
                 num_layers=6, eeg_feature_size=128, fin_feature_size= 128, dropout=0.5,
                 num_classes=2):

        super(EFFusionNet, self).__init__()
        self.face_feature_extractor = FaceFeatureExtractor(feature_size=face_feature_size, pretrain=pretrain)

        self.eeg_feature_extractor = DualSpatialTimeModel(sampling_rate,eeg_channel, s_channel,
                                                          s_kernel, t_kernel, t_channel,
                                                          hidden_dim, kernel_size, bias, dtype,
                                                          num_layers, eeg_feature_size)
        self.EEGPrivate = EEGPrivate(eeg_feature_size, dropout)
        self.FacePrivate = FacePrivate(face_feature_size, dropout)
        self.CommonNet = CommonNet(eeg_feature_size, dropout)
        self.com_fc = nn.Sequential(
            nn.Linear(fin_feature_size//2, fin_feature_size//4),
            nn.Dropout(dropout)
        )

        # self.score_eeg = ScoreNet()
        # self.score_face = ScoreNet()
        # self.score_com = ScoreNet()
        self.score_eeg = ScoreNet(fin_feature_size)
        self.score_generate1 = nn.Sequential(
            nn.Linear(fin_feature_size//4,1)
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(fin_feature_size//4*3, fin_feature_size//8*3),
            nn.Dropout(dropout)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(fin_feature_size//8*3, fin_feature_size//8*3),
            nn.Dropout(dropout),
        )
        if num_classes == 1:
            self.fc = nn.Sequential(
                nn.Linear(fin_feature_size//8*3, num_classes),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fin_feature_size//8*3, num_classes),
                #nn.Sigmoid()
            )
    def forward(self, x):
        result = []
        face_out = self.face_feature_extractor(x[0])
        eeg_out = self.eeg_feature_extractor(x[1])

        # print('face_out',face_out.shape)
        # print('eeg_out', eeg_out.shape)

        eeg_output = self.EEGPrivate(eeg_out)
        face_output = self.FacePrivate(face_out)
        #print('eeg_output', eeg_output.shape)
        #print('face_output', face_output.shape)
        eeg_com_output = self.CommonNet(eeg_out)
        face_com_output = self.CommonNet(face_out)

        result.append(eeg_output)
        result.append(face_output)
        result.append(eeg_com_output)
        result.append(face_com_output)

        features = torch.cat([eeg_com_output, face_com_output],dim=1)
        com_output = self.com_fc(features)

        score_feature = torch.cat([eeg_output,face_output,com_output], dim=1)
        score_feature1 = self.score_eeg(score_feature)

        eeg_score = self.score_generate1(score_feature1)
        face_score = self.score_generate1(score_feature1)
        com_score = self.score_generate1(score_feature1)

        all_output = torch.cat((eeg_score, face_score, com_score), dim=1)
        weights = F.softmax(all_output, dim=1)
        #print(weights)
        eeg = weights[:, 0].unsqueeze(1)*eeg_output
        face = weights[:, 1].unsqueeze(1)*face_output
        com = weights[:, 2].unsqueeze(1)*com_output

        # print('eeg', eeg.shape)
        # print('face', face.shape)
        # print('com', com.shape)

        all_feature = torch.cat((eeg, face, com), dim=1)
        #print('all_feature',all_feature.shape)
        all_feature_fc_1 = self.fc_1(all_feature)
        all_feature_fc_2 = self.fc_2(all_feature_fc_1)
        result.append(all_feature_fc_1)
        result.append(all_feature_fc_2)
        finally_out = self.fc(all_feature_fc_2)
        #print('finally_out',finally_out.shape)
        finally_out = finally_out.squeeze(-1)
        result.append(finally_out)

        return result

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


class EFFusion_feature_level(nn.Module):
    def __init__(self, face_feature_size=128, pretrain=True,
                 eeg_channel=128, s_channel=32,
                 s_kernel=3, t_kernel=5, t_channel=32,
                  hidden_dim=32, kernel_size=(3, 3), bias=True, dtype=torch.cuda,
                 num_layers=3, eeg_feature_size=128, fin_feature_size= 128, dropout=0.2
                 ):

        super(EFFusion_feature_level, self).__init__()
        self.face_feature_extractor = FaceFeatureExtractor(feature_size=face_feature_size, pretrain=pretrain)

        self.eeg_feature_extractor = DualSpatialTimeModel(eeg_channel, s_channel,
                                                          s_kernel, t_kernel, t_channel,
                                                          hidden_dim, kernel_size, bias, dtype,
                                                          num_layers, eeg_feature_size)


        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            #nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #result = []
        face_out = self.face_feature_extractor(x[0])
        #print('face_out.shape',face_out.shape)
        eeg_out = self.eeg_feature_extractor(x[1])
        #print('eeg_out.shape',eeg_out.shape)
        all_feature = torch.cat((face_out, eeg_out), dim=1)
        finally_out = self.fc(all_feature)
        #print('finally_out',finally_out.shape)
        finally_out = finally_out.squeeze(-1)
        #result.append(finally_out)

        return finally_out

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


class decision_net(nn.Module):
    def __init__(self,dropout):
        super(decision_net, self).__init__()
        self.decision_score = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #print('x',x.shape)
        x_score = self.decision_score(x)
        #attention_weights = torch.softmax(x_score, dim=1)

        return x_score


class EFFusionNet_decision(nn.Module):
    def __init__(self, face_feature_size=128, pretrain=True,
                 sampling_rate=128,eeg_channel=32, s_channel=32,
                 s_kernel=3, t_kernel=5, t_channel=32,
                  hidden_dim=32, kernel_size=(3, 3), bias=True, dtype=torch.cuda,
                 num_layers=3, eeg_feature_size=128, fin_feature_size= 128, dropout=0.2
                 ):

        super(EFFusionNet_decision, self).__init__()
        self.face_feature_extractor = FaceFeatureExtractor(feature_size=face_feature_size, pretrain=pretrain)

        self.eeg_feature_extractor = DualSpatialTimeModel(sampling_rate,eeg_channel, s_channel,
                                                          s_kernel, t_kernel, t_channel,
                                                          hidden_dim, kernel_size, bias, dtype,
                                                          num_layers, eeg_feature_size)
        self.EEGPrivate = EEGPrivate(eeg_feature_size, dropout)
        self.FacePrivate = FacePrivate(face_feature_size, dropout)
        self.CommonNet = CommonNet(eeg_feature_size, dropout)
        self.com_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(dropout)
        )

        # self.score_eeg = ScoreNet()
        # self.score_face = ScoreNet()
        # self.score_com = ScoreNet()
        self.score_eeg = ScoreNet()
        self.score_generate1 = nn.Sequential(
            nn.Linear(32,1)
        )

        self.fc = decision_net(dropout)

    def forward(self, x):
        result = []
        face_out = self.face_feature_extractor(x[0])
        eeg_out = self.eeg_feature_extractor(x[1])

        # print('face_out',face_out.shape)
        # print('eeg_out', eeg_out.shape)

        eeg_output = self.EEGPrivate(eeg_out)
        face_output = self.FacePrivate(face_out)
        #print('eeg_output', eeg_output.shape)
        #print('face_output', face_output.shape)
        eeg_com_output = self.CommonNet(eeg_out)
        face_com_output = self.CommonNet(face_out)

        result.append(eeg_output)
        result.append(face_output)
        result.append(eeg_com_output)
        result.append(face_com_output)

        features = torch.cat([eeg_com_output, face_com_output],dim=1)
        com_output = self.com_fc(features)

        score_feature = torch.cat([eeg_output,face_output,com_output], dim=1)
        score_feature1 = self.score_eeg(score_feature)

        eeg_score = self.score_generate1(score_feature1)
        face_score = self.score_generate1(score_feature1)
        com_score = self.score_generate1(score_feature1)

        all_output = torch.cat((eeg_score, face_score, com_score), dim=1)
        weights = F.softmax(all_output, dim=1)
        #print(weights)
        eeg = weights[:, 0].unsqueeze(1)*eeg_output
        face = weights[:, 1].unsqueeze(1)*face_output
        com = weights[:, 2].unsqueeze(1)*com_output

        # print('eeg', eeg.shape)
        # print('face', face.shape)
        # print('com', com.shape)

        #print('all_feature',all_feature.shape)
        eeg_finally_out = self.fc(eeg)
        face_finally_out = self.fc(face)
        com_finally_out = self.fc(com)
        #print('finally_out',finally_out.shape)
        eeg_pred = eeg_finally_out.squeeze(-1)
        face_pred = face_finally_out.squeeze(-1)
        com_pred = com_finally_out.squeeze(-1)
        result.append(eeg_pred)
        result.append(face_pred)
        result.append(com_pred)

        return result

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))