import torch.nn.functional
from data import *
from torch.utils.data import Dataset
import csv
import torch
from sklearn.preprocessing import StandardScaler

"""
将预提取的音频属性转化为特征
"""


def load_audio_attribute(filepath):
    csvfile = filepath[:-4] + '_is09.csv'
    reader = csv.reader(open(csvfile, 'r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    feature = last_line[1: 384 + 1]
    feature = [float(x) for x in feature]
    feature = np.asarray(feature)
    feature = torch.from_numpy(feature)
    feature = torch.unsqueeze(feature, -1)

    scaler = StandardScaler().fit(feature)
    feature = scaler.transform(feature)
    feature = torch.from_numpy(feature)
    feature = torch.squeeze(feature)
    return feature


"""
将预提取的人脸属性转化为特征
"""


def load_image_attribute(imagepath):
    txt = imagepath[:-11] + '.txt'
    attri = []
    with open(txt, 'r') as f:
        for line in f:
            attri.append(int(line.split()[1]))
    attri = np.asarray(attri)
    attri = torch.from_numpy(attri)
    return attri


"""
训练数据加载
"""


class train_data_V2F(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(0)
        print('the number of training: ', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        a, f1, f2, label, a_gender, f1_gender, f2_gender, a_nation, f1_nation, f2_nation = self.data[item]
        label = int(label)
        a_gender = int(a_gender)
        f1_gender = int(f1_gender)
        f2_gender = int(f2_gender)
        a_nation = int(a_nation)
        f1_nation = int(f1_nation)
        f2_nation = int(f2_nation)

        a_attribute = load_audio_attribute(a)
        f1_attribute = load_image_attribute(f1)
        f2_attribute = load_image_attribute(f2)

        a = load_audio(a)
        a = torch.from_numpy(a)
        a = torch.unsqueeze(a, dim=0)

        f1 = TransformToPIL(f1)
        f1 = target_transform_train(f1)

        f2 = TransformToPIL(f2)
        f2 = target_transform_train(f2)

        face_m = 0
        audio_m = 1

        return a, a_attribute, a_gender, a_nation, f1, f1_attribute, f1_gender, f1_nation, f2, f2_attribute, f2_gender, f2_nation, label, face_m, audio_m

    def __len__(self):
        return len(self.data)


"""
测试数据加载
"""


class test_data_V2F(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(1)
        print('the number of test: ', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        a, f1, f2, label = self.data[item]
        label = int(label)

        a_attribute = load_audio_attribute(a)
        f1_attribute = load_image_attribute(f1)
        f2_attribute = load_image_attribute(f2)

        a = load_audio(a)
        a = torch.from_numpy(a)
        a = torch.unsqueeze(a, dim=0)

        f1 = TransformToPIL(f1)
        f1 = target_transform_train(f1)

        f2 = TransformToPIL(f2)
        f2 = target_transform_train(f2)

        return a, a_attribute, f1, f1_attribute, f2, f2_attribute, label

    def __len__(self):
        return len(self.data)
