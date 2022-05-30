import warnings
from utils import *
from model import *
import numpy as np

warnings.filterwarnings('ignore')

"""
模型预加载
"""


def load(feature, V, D, C):
    states = torch.load('.pkl')
    feature.load_state_dict(states['feature'])
    V.load_state_dict(states['V'])
    D.load_state_dict(states['D'])
    C.load_state_dict(states['C'])
    return feature, V, D, C


def train():
    feature = feature_extractor()  # 特征提取网络
    fusion_audio = CrossTransformerEncoderLayer()  # 音频特征融合
    fusion_visual = Visual()  # 人脸特征融合
    gender = Gender()  # 性别分类器
    nation = Nation()  # 国籍分类器
    vae = VAE()  # 变分自编码器
    dis = discriminator()  # 鉴别器
    cls = Class()  # id分类器
    loss_fc = nn.CrossEntropyLoss()  # 交叉熵损失
    Rank_loss = lift_struct(1.2, 1, 0.3)  # 度量学习损失
    cuda = True if torch.cuda.is_available() else False
    feature, vae, dis, cls = load(feature, vae, dis, cls)

    if cuda:
        feature = feature.to('cuda')
        vae = vae.to('cuda')
        dis = dis.to('cuda')
        fusion_audio = fusion_audio.to('cuda')
        fusion_visual = fusion_visual.to('cuda')
        nation = nation.to('cuda')
        gender = gender.to('cuda')
        Rank_loss = Rank_loss.to('cuda')
        cls = cls.to('cuda')
        loss_fc = loss_fc.to('cuda')

    batch_size = 50
    acc_best = 0.0
    train_data = train_data_F2V()
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=10)  # 训练数据加载
    optimizer_G = torch.optim.Adam(  # 生成器参数
        [
            {"params": feature.parameters(), "lr": 5e-2},
            {"params": vae.parameters(), "lr": 1e-2},
            {"params": cls.parameters(), "lr": 1e-2},
            {"params": gender.parameters(), "lr": 5e-3},
            {"params": nation.parameters(), "lr": 5e-3},
        ]
    )
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=5e-3, betas=(0.5, 0.999))  # 鉴别器参数

    for epoch in range(100):
        adjust_lr(optimizer_G, epoch)  # 动态调节学习率
        feature.train()
        vae.train()
        dis.train()
        nation.train()
        gender.train()
        cls.train()
        train_count = 0.0
        audio_count = 0.0
        face_count = 0.0
        total_train = 0.0
        for i, data in enumerate(data_loader):
            f, a1, a2, label, f_gender, a1_gender, a2_gender, f_nation, a1_nation, a2_nation, \
            f_attribute, a1_attribute, a2_attribute, face_m, audio_m = data  # 数据加载
            f = f.to('cuda')
            f_gender = f_gender.to('cuda')
            f_nation = f_nation.to('cuda')
            f_attribute = f_attribute.to('cuda')
            a1 = a1.to('cuda')
            a1_gender = a1_gender.to('cuda')
            a1_nation = a1_nation.to('cuda')
            a1_attribute = a1_attribute.to('cuda')
            a2 = a2.to('cuda')
            a2_attribute = a2_attribute.to('cuda')
            a2_gender = a2_gender.to('cuda')
            a2_nation = a2_nation.to('cuda')
            label = label.to('cuda')
            face_m = face_m.to('cuda')
            audio_m = audio_m.to('cuda')
            total_train += f.size(0)

            f, a1, a2 = feature(f, a1, a2)  # 先进行特征提取
            """
            通过VAE来减小类内差异性
            """
            recon_f, f, encoder_f, z_f, mu_f, log_var_f = vae(f)
            recon_a1, a1, encoder_a1, z_a1, mu_a1, log_var_a1 = vae(a1)
            recon_a2, a2, encoder_a2, z_a2, mu_a2, log_var_a2 = vae(a2)
            """
            私有属性融合
            """
            encoder_f = fusion_visual(encoder_f, f_attribute)
            encoder_a1 = fusion_audio(encoder_a1, a1_attribute)
            encoder_a2 = fusion_audio(encoder_a2, a2_attribute)

            for p1 in dis.parameters():
                p1.requires_grad = True
            for p2 in feature.parameters():
                p2.requires_grad = False
            for p3 in vae.parameters():
                p3.requires_grad = False
            for p4 in cls.parameters():
                p4.requires_grad = False
            for p5 in nation.parameters():
                p5.requires_grad = False
            for p6 in gender.parameters():
                p6.requires_grad = False

            for k in range(5):
                for p in dis.parameters():
                    p.data.clamp_(-0.01, 0.01)
                out_f, out_a1, out_a2 = dis(z_f, z_a1, z_a2)

                loss_d = 2 * loss_fc(out_f, face_m) + loss_fc(out_a1, audio_m) + loss_fc(out_a2, audio_m)
                optimizer_D.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_D.step()
            face_count += label_acc(out_f, face_m)
            audio_count += label_acc(out_a1, audio_m) + label_acc(out_a2, audio_m)

            for p1 in dis.parameters():
                p1.requires_grad = False
            for p2 in feature.parameters():
                p2.requires_grad = True
            for p3 in vae.parameters():
                p3.requires_grad = True
            for p4 in cls.parameters():
                p4.requires_grad = True
            for p5 in nation.parameters():
                p5.requires_grad = True
            for p6 in gender.parameters():
                p6.requires_grad = False

            out_f, out_a1, out_a2 = dis(z_f, z_a1, z_a2)
            nation_f, nation_a1, nation_a2 = nation(encoder_f, encoder_a1, encoder_a2)  # 性别分类结果
            gender_f, gender_a1, gender_a2 = gender(encoder_f, encoder_a1, encoder_a2)  # 国籍分类结果
            predict = cls(encoder_f, encoder_a1, encoder_a2)  # id分类结果

            loss_nation = 2 * loss_fc(nation_f, f_nation) + loss_fc(nation_a1, a1_nation) + loss_fc(nation_a2,
                                                                                                    a2_nation)
            loss_gender = 2 * loss_fc(gender_f, f_gender) + loss_fc(gender_a1, a1_gender) + loss_fc(gender_a2,
                                                                                                    a2_gender)
            """
            对分类结果进行损失计算
            """
            loss_g = 2 * loss_fc(out_f, audio_m) + loss_fc(out_a1, face_m) + loss_fc(out_a2, face_m)
            loss_cls = loss_fc(predict, label)
            """
            计算重构误差
            """
            loss_f = loss_fuction(recon_f, f, mu_f, log_var_f)
            loss_a = loss_fuction(recon_a1, a1, mu_a1, log_var_a1) + loss_fuction(recon_a2, a2, mu_a2, log_var_a2)
            loss_recon = 2 * loss_f + loss_a
            """
            度量学习损失
            """
            loss_m = compute_metric(encoder_f, encoder_a1, encoder_a2, label, Rank_loss)
            loss_total = loss_recon + loss_g + loss_cls + loss_nation + loss_gender + loss_m
            train_count += label_acc(predict, label)

            if i % 10 == 0:
                print(epoch, i, ' C ', loss_cls.item(), ' G ', loss_g.item(), ' D ', loss_d.item(), 'recon',
                      loss_recon.item(), 'Nation', loss_nation.item(), 'Gender', loss_gender.item(), 'M', loss_m.item())
                if train_count != 0:
                    print('counts =', train_count)

            optimizer_G.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer_G.step()

        face_acc = face_count / total_train
        audio_acc = audio_count / (2 * total_train)
        acc = train_count / total_train

        print('epoch:', epoch, 'F2V training acc :', acc)
        print('Audio acc : ', audio_acc, 'Face acc : ', face_acc)
        acc_best = eval(feature, fusion_audio, fusion_visual, vae, cls, dis, epoch, acc_best)
    print('training over')
    print('acc_best= ', acc_best)


if __name__ == '__main__':
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train()
