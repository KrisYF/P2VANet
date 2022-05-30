import torch.nn.functional as F
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn


class lift_struct(nn.Module):
    def __init__(self, alpha, multi, margin):
        super(lift_struct, self).__init__()
        self.alpha = alpha
        self.multi = multi
        self.margin = margin
        self.loss = nn.TripletMarginLoss(self.margin)

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


def compute_metric(f, a1, a2, label, loss_fuc):
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


def eval(feature, fusion_audio, fusion_visual, vae, cls, dis, epoch, acc_best):
    test_data = test_data_F2V()
    data_loader = DataLoader(test_data, batch_size=50, shuffle=False, num_workers=8)
    feature.eval()
    cls.eval()
    test_total = 0.0
    test_count = 0.0
    for index, data in enumerate(data_loader):
        f, f_attribute, a1, a1_attribute, a2, a2_attribute, label = data
        f = f.to('cuda')
        f_attribute = f_attribute.to('cuda')
        a1 = a1.to('cuda')
        a1_attribute = a1_attribute.to('cuda')
        a2 = a2.to('cuda')
        a2_attribute = a2_attribute.to('cuda')
        label = label.to('cuda')

        test_total += f.size(0)

        f, a1, a2 = feature(f, a1, a2)
        recon_f, f, encoder_f, z_f, mu_f, log_var_f = vae(f)
        recon_a1, a1, encoder_a1, z_a1, mu_a1, log_var_a1 = vae(a1)
        recon_a2, a2, encoder_a2, z_a2, mu_a2, log_var_a2 = vae(a2)

        encoder_f = fusion_visual(encoder_f, f_attribute)
        encoder_a1 = fusion_audio(encoder_a1, a1_attribute)
        encoder_a2 = fusion_audio(encoder_a2, a2_attribute)

        predict = cls(encoder_f, encoder_a1, encoder_a2)
        test_count += label_acc(predict, label)
    acc = test_count / test_total
    if acc > acc_best:
        acc_best = acc
        state = {
            'feature': feature.state_dict(),
            'V': vae.state_dict(),
            'D': dis.state_dict(),
            'C': cls.state_dict()
        }
        name = 'F2V' + str(epoch) + '_' + str(acc) + '.pkl'
        torch.save(state, name)
    print('current best acc: ', acc, 'best acc: ', acc_best)
    return acc_best
