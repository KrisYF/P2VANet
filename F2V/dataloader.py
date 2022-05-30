import torch
import torch.nn.functional
from torch.utils.data import Dataset
from data import *
import csv
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


class train_data_F2V(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(0)
        print('the number of training: ', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        f, a1, a2, label, f_gender, a1_gender, a2_gender, f_nation, a1_nation, a2_nation = self.data[item]
        label = int(label)
        f_nation = int(f_nation)
        a1_nation = int(a1_nation)
        a2_nation = int(a2_nation)
        f_gender = int(f_gender)
        a1_gender = int(a1_gender)
        a2_gender = int(a2_gender)

        f_attribute = load_image_attribute(f)
        a1_attribute = load_audio_attribute(a1)
        a2_attribute = load_audio_attribute(a2)

        a1 = load_audio(a1)
        a1 = torch.from_numpy(a1)
        a1 = torch.unsqueeze(a1, dim=0)

        a2 = load_audio(a2)
        a2 = torch.from_numpy(a2)
        a2 = torch.unsqueeze(a2, dim=0)

        f = TransformToPIL(f)
        f = target_transform_train(f)

        face_m = 0
        audio_m = 1

        return f, a1, a2, label, f_gender, a1_gender, a2_gender, f_nation, a1_nation, a2_nation, f_attribute, a1_attribute, a2_attribute, face_m, audio_m

    def __len__(self):
        return len(self.data)


"""
测试数据加载
"""


class test_data_F2V(Dataset):
    def __init__(self):
        Fusion_list = get_anchor_audio(1)
        print('the number of test: ', len(Fusion_list))
        self.data = Fusion_list

    def __getitem__(self, item):
        f, a1, a2, label = self.data[item]
        label = int(label)

        f_attribute = load_image_attribute(f)
        a1_attribute = load_audio_attribute(a1)
        a2_attribute = load_audio_attribute(a2)

        a1 = load_audio(a1)
        a1 = torch.from_numpy(a1)
        a1 = torch.unsqueeze(a1, dim=0)

        a2 = load_audio(a2)
        a2 = torch.from_numpy(a2)
        a2 = torch.unsqueeze(a2, dim=0)

        f = TransformToPIL(f)
        f = target_transform_train(f)

        return f, f_attribute, a1, a1_attribute, a2, a2_attribute, label

    def __len__(self):
        return len(self.data)
