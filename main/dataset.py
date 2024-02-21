"""
DEAP and MAHNOB datasets
"""
import torch
import os
from PIL import Image
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T
import io
import zipfile
import cv2
import random

class DEAP20(data.Dataset):
    def __init__(self, modal, subject, kind, indices, label):
        self.modal = modal

        self.kind = kind
        self.subject = subject
        self.label = label
        self.bio_path = f'/data/home/.data/jiangjiewei/peimengjie/projects/lz/CY/dataset/bio/'
        self.label_path = f'/data/home/.data/jiangjiewei/peimengjie/projects/lz/CY/dataset/labels/'
        self.face_path = f'/data/home/.data/jiangjiewei/peimengjie/projects/lz/CY/dataset/faces'
        #self.labels = pd.read_csv(self.label_path+'participant_ratings.csv')
        self.labels = pd.read_csv(self.label_path+'participant_ratings.csv')
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
        self.sub_trial_seg = []
        num_trials = int(deap_indices_dict[self.subject] // 60)
        train_trials = random.sample(range(1, num_trials + 1), deap_indices_dict[self.subject] // 60 // 2)
        val_trials = [trial for trial in range(1, num_trials + 1) if trial not in train_trials]
        if self.kind == 'train':
            for trial in train_trials:
                for seg in range(1, 61):
                    self.sub_trial_seg.append((self.subject, trial, seg))
        if self.kind == 'val':
            for trial in val_trials:
                for seg in range(1, 61):
                    self.sub_trial_seg.append((self.subject, trial, seg))
        len_train = len(self.sub_trial_seg)
        t_indices = list(range(len_train))
        if self.kind == 'train':
            self.indices = t_indices
        if self.kind == 'val':
            self.indices = t_indices

    def __getitem__(self, i):
        index = self.indices[i]
        subject, trial, segment = self.sub_trial_seg[index]
        prex = 's' + (str(self.subject) if self.subject > 9 else '0' + str(self.subject)) + '/s' + (
            str(self.subject) if self.subject > 9 else '0' + str(self.subject)) + '_trial' + (
                   str(trial) if trial > 9 else '0' + str(trial)) + '/s' + (
                   str(self.subject) if self.subject > 9 else '0' + str(self.subject)) + '_trial' + (
                   str(trial) if trial > 9 else '0' + str(trial))
        transform = T.Compose([T.ToPILImage(),
                               T.Resize((64, 64)),
                               T.ToTensor()])
        face_data = []
        for n in range(1, 6):
            x1 = self.face_path + '/' + prex + f'_{(segment - 1) * 5 + n}.png'
            img = cv2.imread(x1)
            frame_array = transform(img)
            frame_array = frame_array.view(1, 3, 64, 64)
            face_data.append(frame_array)
        face_data = torch.cat(face_data, dim=0)
        #x2 = self.bio_zip.read(f's{self.subject}/{self.subject}_{trial}_{segment}.npy')
        x2 = self.bio_path + '/' + f's{self.subject}/{self.subject}_{trial}_{segment}.npy'
        bio_data = torch.tensor(np.load(x2)).float()

        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'faceeeg':
            data = (face_data, bio_data[:32])
        if self.label == '4class':
            valence1 = \
                self.labels[(self.labels['Participant_id'] == subject) & (self.labels['Trial'] == trial)][
                    'Valence'].iloc[0]
            arousal1 = \
                self.labels[(self.labels['Participant_id'] == subject) & (self.labels['Trial'] == trial)][
                    'Arousal'].iloc[0]
            if valence1 > 5 and arousal1 > 5:
                label = 0
            elif valence1 > 5 and arousal1 <= 5:
                label = 1
            elif valence1 <= 5 and arousal1 > 5:
                label = 2
            elif valence1 <= 5 and arousal1 <= 5:
                label = 3
            return data, label

        elif self.label == 'valence':
            valence = 0 if \
                self.labels[(self.labels['Participant_id'] == subject) & (self.labels['Trial'] == trial)][
                    'Valence'].iloc[0] < 5 else 1
            return data, valence
        elif self.label == 'arousal':
            arousal = 0 if \
                self.labels[(self.labels['Participant_id'] == subject) & (self.labels['Trial'] == trial)][
                    'Arousal'].iloc[0] < 5 else 1
            return data, arousal

    def __len__(self):
        return len(self.indices)


class MAHNOB20(data.Dataset):
    def __init__(self, modal, subject, kind, indices, label):
        self.modal = modal
        self.subject = subject
        self.kind = kind
        self.label = label
        self.bio_path = f'/data/home/.data/jiangjiewei/peimengjie/projects/lz/CY/mahnob/bio/{subject}'
        self.label_path = '/data/home/.data/jiangjiewei/peimengjie/projects/lz/CY/mahnob/labels/mahnob_labels.npy'
        self.face_path = f'/data/home/.data/jiangjiewei/peimengjie/projects/lz/CY/mahnob/faces/{subject}'
        self.labels = np.load(self.label_path)
        self.label_dict = dict()
        for label in self.labels:
            self.label_dict[label[0]] = label[1:]
        self.trials = []
        face_entries = os.listdir(self.face_path)
        for i in face_entries:
            self.trials.append(int(i))
        num_trials = len(self.trials)
        train_trials = random.sample(self.trials, num_trials // 5*4)
        val_trials = [trial for trial in self.trials if trial not in train_trials]
        self.trial_seg = []
        if self.kind == 'train':
            for trial in train_trials:
                for seg in range(1, self.label_dict[trial][1]+1):
                    self.trial_seg.append((trial, seg))
        if self.kind == 'val':
            for trial in val_trials:
                for seg in range(1, self.label_dict[trial][1]+1):
                    self.trial_seg.append((trial, seg))
        len_train = len(self.trial_seg)
        t_indices = list(range(len_train))
        if kind == 'train':
            self.indices = t_indices
        if kind == 'val':
            self.indices = t_indices


    def __getitem__(self, i):
        index = self.indices[i]
        trial, segment = self.trial_seg[index]
        transform = T.Compose([T.ToPILImage(),
                               T.Resize((64, 64)),
                               T.ToTensor()])
        face_data = []
        for n in range(1, 6):
            x1 = self.face_path + '/' + str(trial) + '/' f'{trial}_{(segment - 1) * 5 + n}.png'
            try:
                img = cv2.imread(x1)
                if img is None:
                    raise Exception(f"Image {x1} not found.")
                frame_array = transform(img)
                frame_array = frame_array.view(1, 3, 64, 64)
                face_data.append(frame_array)
            except Exception as e:
                print(f"Error loading image: {e}")
        # 将face_data中的所有张量拼接成一个大张量
        if len(face_data) > 0:
            face_data = torch.cat(face_data, dim=0)
            #print("Images loaded successfully.")
        else:
            print("No valid images found.")

        x2 = self.bio_path + '/' + f'{self.subject}_{trial}_{segment}.npy'
        bio_data = torch.tensor(np.load(x2)).float()

        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'faceeeg':
            data = (face_data, bio_data[:32])

        if self.label == 'valence':
            valence = 0 if self.label_dict[trial][2] < 5 else 1
            return data, valence
        elif self.label == 'arousal':
            arousal = 0 if self.label_dict[trial][3] < 5 else 1
            return data, arousal
        elif self.label == '4class':
            if self.label_dict[trial][2] <= 5 and self.label_dict[trial][3] <= 5:
                label = 3
            elif self.label_dict[trial][2] <= 5 and self.label_dict[trial][3] > 5:
                label = 2
            elif self.label_dict[trial][2] > 5 and self.label_dict[trial][3] <= 5:
                label = 1
            elif self.label_dict[trial][2] > 5 and self.label_dict[trial][3] > 5:
                label = 0

            return data, label

    def __len__(self):
        return len(self.indices)