import torch.nn.functional as F
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn


def loss_fuction(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = recon_loss + kld_loss

    return loss


def label_acc(out, label):
    label = label.to('cuda')
    _, predicts = torch.max(out.data, 1)
    correct = (predicts == label).sum().item()
    return correct


def adjust_lr(optimizer, epoch):
    if epoch >= 30 and epoch % 10 == 0:
        for p in optimizer.param_groups:
            p['lr'] = p['lr'] * 0.1


def computer_metric(f, a1, a2, label, loss_fuc):
    label = label.cpu().numpy()
    mod0 = np.where(label == 0)
    mod1 = np.where(label == 1)
    f_0 = f[mod0[0]]
    f_1 = f[mod1[0]]
    a1_0 = a1[mod0[0]]
    a1_1 = a1[mod1[0]]
    a2_0 = a2[mod0[0]]
    a2_1 = a2[mod1[0]]
    n_0 = []
    n_1 = []
    n_0.append(a2_0)
    n_1.append(a1_1)
    loss = loss_fuc(f_0, a1_0, n_0) + loss_fuc(f_1, a2_1, n_1)
    return loss


class lift_struct(nn.Module):
    def __init__(self, alpha, multi):
        super(lift_struct, self).__init__()
        self.alpha = alpha
        self.multi = multi

    def forward(self, anchor, positive, neglist):
        batch = anchor.size(0)
        D_ij = torch.pairwise_distance(anchor, positive)
        D_in = 0
        D_jn = 0
        for i in range(self.multi):
            a = torch.pairwise_distance(anchor, neglist[i])
            D_in += torch.exp(self.alpha - a)
            b = torch.pairwise_distance(positive, neglist[i])
            D_jn += torch.exp(self.alpha - b)
        D_n = D_in + D_jn
        J = torch.log(D_n) + D_ij
        J = torch.clamp(J, min=0)
        loss = J.sum() / (2 * batch)
        return loss


def eval(feature, fusion_audio, fusion_visual, vae, cls, dis, epoch, acc_best):
    test_data = test_data_V2F()
    data_loader = DataLoader(test_data, batch_size=50, shuffle=False, num_workers=10)
    feature.eval()
    cls.eval()
    test_total = 0.0
    test_count = 0.0
    for index, data in enumerate(data_loader):
        a, a_attribute, f1, f1_attribute, f2, f2_attribute, label = data
        a = a.to('cuda')
        a_attribute = a_attribute.to('cuda')
        f1 = f1.to('cuda')
        f1_attribute = f1_attribute.to('cuda')
        f2 = f2.to('cuda')
        f2_attribute = f2_attribute.to('cuda')
        label = label.to('cuda')
        test_total += a.size(0)

        a, f1, f2 = feature(a, f1, f2)
        recon_a, a, encoder_a, z_a, mu_a, log_var_a = vae(a)
        recon_f1, f1, encoder_f1, z_f1, mu_f1, log_var_f1 = vae(f1)
        recon_f2, f2, encoder_f2, z_f2, mu_f2, log_var_f2 = vae(f2)

        encoder_a = fusion_audio(encoder_a, a_attribute)
        encoder_f1 = fusion_visual(encoder_f1, f1_attribute)
        encoder_f2 = fusion_visual(encoder_f2, f2_attribute)

        predict = cls(encoder_a, encoder_f1, encoder_f2)
        test_count += label_acc(predict, label)
    acc = test_count / test_total
    if acc > acc_best:
        acc_best = acc
        state = {
            'feature': feature.state_dict(),
            'V': vae.state_dict(),
            'D': dis.state_dict(),
            'C': cls.state_dict(),
            'A': fusion_audio.state_dict(),
            'I': fusion_visual.state_dict()
        }
        name = 'V2F' + str(epoch) + '_' + str(acc) + '.pkl'
        torch.save(state, name)

    print('current best acc: ', acc, 'best acc: ', acc_best)
    return acc_best
