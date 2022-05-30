import torchvision.transforms as transforms
import librosa
import numpy as np
from PIL import Image

"""
训练图像预处理
"""


def target_transform_train(PILImg):
    transf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transf(PILImg)


"""
测试图像预处理
"""


def target_transform_test(PILImg):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transf(PILImg)


"""
将图像裁剪为224*224大小
"""


def TransformToPIL(PILimage):
    RealPILimage = Image.open(PILimage).convert("RGB")
    RealPILimage = RealPILimage.resize((224, 224))
    return RealPILimage


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


"""
音频加载
"""


def load_audio(audiopath):
    y, sr = librosa.load(audiopath)
    y = y - y.mean()
    y = preemphasis(y)
    y = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    a = np.max(y)
    b = np.min(y)
    k = 2 / (a - b)
    y = -1 + (y - b) * k
    y = np.resize(y, [224, 125])
    return y


"""
获取训练和测试数据
"""


def get_anchor_audio(flag):
    FusionList_anchor_audio = []
    if flag == 0:
        with open('train.txt', 'r') as f:
            for line in f:
                str = line.split()
                FusionList_anchor_audio.append(
                    (str[0],) + (str[1],) + (str[2],) + (str[3],) + (str[4],) + (str[5],) + (str[6],) + (str[7],) + (
                        str[8],) + (str[9],))  # str为[音频a, 人脸f1, 人脸f2, 匹配标签，a的性别，f1的性别，f2的性别，a的国籍，f1的国籍，f2的国籍
    elif flag == 1:
        with open('test.txt', 'r') as f:
            for line in f:
                str = line.split()
                FusionList_anchor_audio.append((str[0],) + (str[1],) + (str[2],) + (str[3],))

    return FusionList_anchor_audio
