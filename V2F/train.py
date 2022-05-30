import warnings
from utils import *
from model import *

warnings.filterwarnings('ignore')

"""
模型预加载
"""


def load(feature, V, D, C):
    states = torch.load('')
    feature.load_state_dict(states['feature'])
    V.load_state_dict(states['V'])
    D.load_state_dict(states['D'])
    C.load_state_dict(states['C'])
    return feature, V, D, C


def train():
    feature = FeatureExtractor()  # 特征提取网络
    fusion_audio = CrossTransformerEncoderLayer()  # 音频特征融合
    fusion_visual = Visual()  # 人脸特征融合
    gender = Gender()  # 性别分类器
    nation = Nation()  # 国籍分类器
    vae = VAE()  # 变分自编码器
    dis = Discriminator()  # 鉴别器
    cls = Class()  # id分类器
    loss_fc = nn.CrossEntropyLoss()  # 交叉熵损失
    Rank_loss = lift_struct(1.2, 1)  # 度量学习损失
    cuda = True if torch.cuda.is_available() else False
    # feature, vae, dis, cls = load(feature, vae, dis, cls)

    if cuda:
        feature = feature.to('cuda')
        fusion_audio = fusion_audio.to('cuda')
        fusion_visual = fusion_visual.to('cuda')
        gender = gender.to('cuda')
        nation = nation.to('cuda')
        vae = vae.to('cuda')
        dis = dis.to('cuda')
        cls = cls.to('cuda')
        loss_fc = loss_fc.to('cuda')
        Rank_loss = Rank_loss.to('cuda')

    batch_size = 50  # batch设为50
    acc_best = 0.0
    train_data = train_data_V2F()
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=10)  # 训练数据加载
    optimizer_g = torch.optim.Adam(  # 生成器参数
        [
            {"params": feature.parameters(), "lr": 5e-2},
            {"params": vae.parameters(), "lr": 1e-2},
            {"params": cls.parameters(), "lr": 1e-2},
            {"params": gender.parameters(), "lr": 5e-3},
            {"params": nation.parameters(), "lr": 5e-2}
        ]
    )
    optimizer_d = torch.optim.Adam(dis.parameters(), lr=5e-3, betas=(0.5, 0.999))  # 鉴别器参数

    for epoch in range(100):
        adjust_lr(optimizer_g, epoch)  # 动态调节学习率
        feature.train()
        gender.train()
        nation.train()
        vae.train()
        dis.train()
        cls.train()
        train_count = 0.0
        audio_count = 0.0
        face_count = 0.0
        total_train = 0.0
        for i, data in enumerate(data_loader):
            a, a_attribute, a_gender, a_nation, f1, f1_attribute, f1_gender, f1_nation, f2, f2_attribute, f2_gender, \
            f2_nation, label, face_m, audio_m = data  # 数据加载
            a = a.to('cuda')
            a_attribute = a_attribute.to('cuda')
            a_gender = a_gender.to('cuda')
            a_nation = a_nation.to('cuda')
            f1 = f1.to('cuda')
            f1_attribute = f1_attribute.to('cuda')
            f1_gender = f1_gender.to('cuda')
            f1_nation = f1_nation.to('cuda')
            f2 = f2.to('cuda')
            f2_attribute = f2_attribute.to('cuda')
            f2_gender = f2_gender.to('cuda')
            f2_nation = f2_nation.to('cuda')
            label = label.to('cuda')
            face_m = face_m.to('cuda')
            audio_m = audio_m.to('cuda')
            total_train += a.size(0)

            a, f1, f2 = feature(a, f1, f2)  # 先进行特征提取

            """
            通过VAE来减小类内差异性
            """
            recon_a, a, encoder_a, z_a, mu_a, log_var_a = vae(a)
            recon_f1, f1, encoder_f1, z_f1, mu_f1, log_var_f1 = vae(f1)
            recon_f2, f2, encoder_f2, z_f2, mu_f2, log_var_f2 = vae(f2)

            """
            私有属性融合
            """
            encoder_a = fusion_audio(encoder_a, a_attribute)
            encoder_f1 = fusion_visual(encoder_f1, f1_attribute)
            encoder_f2 = fusion_visual(encoder_f2, f2_attribute)

            for p1 in dis.parameters():
                p1.requires_grad = True
            for p2 in feature.parameters():
                p2.requires_grad = False
            for p3 in vae.parameters():
                p3.requires_grad = False
            for p4 in cls.parameters():
                p4.requires_grad = False
            for p5 in gender.parameters():
                p5.requires_grad = False
            for p6 in nation.parameters():
                p6.requires_grad = False

            for k in range(5):
                for p in dis.parameters():
                    p.data.clamp_(-0.01, 0.01)
                out_a, out_f1, out_f2 = dis(z_a, z_f1, z_f2)

                loss_d = 2 * loss_fc(out_a, audio_m) + loss_fc(out_f1, face_m) + loss_fc(out_f2, face_m)  # 鉴别器鉴别
                optimizer_d.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()
            audio_count += label_acc(out_a, audio_m)
            face_count += label_acc(out_f1, face_m) + label_acc(out_f2, face_m)

            for p1 in dis.parameters():
                p1.requires_grad = False
            for p2 in feature.parameters():
                p2.requires_grad = True
            for p3 in vae.parameters():
                p3.requires_grad = True
            for p4 in cls.parameters():
                p4.requires_grad = True
            for p5 in gender.parameters():
                p5.requires_grad = True
            for p6 in nation.parameters():
                p6.requires_grad = True

            out_a, out_f1, out_f2 = dis(z_a, z_f1, z_f2)
            gender_a, gender_f1, gender_f2 = gender(encoder_a, encoder_f1, encoder_f2)  # 性别分类结果
            nation_a, nation_f1, nation_f2 = nation(encoder_a, encoder_f1, encoder_f2)  # 国籍分类结果
            predict = cls(encoder_a, encoder_f1, encoder_f2)  # id分类结果
            """
            对分类结果进行损失计算
            """
            loss_g = 2 * loss_fc(out_a, face_m) + loss_fc(out_f1, audio_m) + loss_fc(out_f2, audio_m)
            loss_gender = 2 * loss_fc(gender_a, a_gender) + loss_fc(gender_f1, f1_gender) + loss_fc(gender_f2,
                                                                                                    f2_gender)
            loss_nation = 2 * loss_fc(nation_a, a_nation) + loss_fc(nation_f1, f1_nation) + loss_fc(nation_f2,
                                                                                                    f2_nation)
            loss_cls = loss_fc(predict, label)

            """
            计算重构误差
            """
            loss_a = loss_fuction(recon_a, a, mu_a, log_var_a)
            loss_f = loss_fuction(recon_f1, f1, mu_f1, log_var_f1) + loss_fuction(recon_f2, f2, mu_f1, log_var_f2)
            loss_recon = 2 * loss_a + loss_f

            """
            度量学习损失
            """
            loss_m = computer_metric(encoder_a, encoder_f1, encoder_f2, label, Rank_loss)

            loss_total = loss_recon + loss_g + loss_cls + loss_m + 0.5 * loss_gender + 0.2 * loss_nation
            train_count += label_acc(predict, label)

            if i % 10 == 0:
                print(epoch, i, ' C ', loss_cls.item(), ' G ', loss_g.item(), ' D ', loss_d.item(), 'recon',
                      loss_recon.item(), 'M', loss_m.item(), 'Gender', loss_gender.item(), 'Nation', loss_nation.item())
                if train_count != 0:
                    print('counts =', train_count)

            optimizer_g.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer_g.step()

        audio_acc = audio_count / total_train
        face_acc = face_count / (2 * total_train)
        acc = train_count / total_train
        print('epoch:', epoch, 'V2F training acc :', acc)
        print('Audio acc : ', audio_acc, 'Face acc : ', face_acc)
        acc_best = eval(feature, fusion_audio, fusion_visual, vae, cls, dis, epoch, acc_best)  # 测试
    print('training over')
    print('acc_best= ', acc_best)


if __name__ == '__main__':
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train()
