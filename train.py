import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import model as m
from data import get_dataloader, train_pipeline, val_pipeline, dv
from tqdm import tqdm
import sklearn.metrics as skm
from types import SimpleNamespace
import os

def qprint(x): 
    if not cfg.quiet: print(x)

def train_model(cfg):
    model = m.to(cfg.device)
    model.dv_prob = cfg.dv_prob
    model.dv_weight = cfg.dv_weight
    val_accs = [0., 0., 0.]
    best_acc = np.sum(val_accs)
    train_data = get_dataloader('train', 0, itrans=train_pipeline, batch_size=64)
    val_datas = get_dataloader('val', list(range(3)), batch_size=128)#, list(range(3)), itrans=val_pipeline, batch_size=64)
    loss_function = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    pbar = tqdm(range(cfg.max_epochs))
    loss = np.nan
    dv_generator = dv()
    for ep in pbar:
        model.train()
        for bidx, (img, lbl) in enumerate(train_data):
            pbar.set_description(desc='Epoch %i, batch %i/%i, loss: %.1f, acc: (0) %.3f (1) %.3f (2) %.3f' % (ep, bidx, len(train_data), loss, val_accs[0], val_accs[1], val_accs[2]))
            img = img.to(cfg.device)
            deep_feat = dv_generator(bs=len(img)).to(cfg.device)
            model.zero_grad()
            out = model(img, deep_feat)
            loss = loss_function(out, lbl.to(cfg.device))
            loss.backward()
            optimizer.step()

        model.eval()
        for olidx in range(3):
            preds = []
            lbls = []
            val_data = val_datas[olidx]
            for bidx, (img, lbl) in enumerate(val_data):
                img = img.to(cfg.device)
                pbar.set_description(desc='Epoch %i, val%i batch %i/%i, acc: (0) %.3f (1) %.3f (2) %.3f' % (ep, olidx, bidx, len(val_data), val_accs[0], val_accs[1], val_accs[2]))
                with torch.no_grad():
                    pred = np.argmax(model(img).detach().cpu().numpy(), axis=1).astype(int)
                    [preds.append(x) for x in pred]
                    lbl = np.array([x.item() for x in lbl]).astype(int)
                    [lbls.append(x) for x in lbl]
            preds, lbls = np.array(preds), np.array(lbls)
            val_accs[olidx] = skm.accuracy_score(np.ravel(preds), np.ravel(lbls))
        if np.sum(val_accs) > best_acc:
            torch.save(model, os.path.join(cfg.save_dir, 'best.pt'))
            best_acc = np.sum(val_accs)
    torch.save(model, os.path.join(cfg.save_dir, 'last.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='device to use', type=str, default='5')
    parser.add_argument('--lr', help='learning rate', type=float, default=5e-5)
    parser.add_argument('--dv-prob', help='deep feature vector probability (alpha in paper)', type=float, default=0.9)
    parser.add_argument('--dv-weight', help='deep feature addition weight (beta in paper)', type=float, default=1.0)
    parser.add_argument('--save-dir', help='save directory for model checkpoint/logs', type=str, default='')
    parser.add_argument('--max-epochs', help='maximum number of epochs', type=int, default=20)
    parser.add_argument('--quiet', help='suppress intermediate details', action='store_true')
    cfg = vars(parser.parse_args())
    print(cfg)
    cfg['device'] = 'cuda:%s' % cfg['device']
    cfg['momentum'] = 0.9
    cfg['weight_decay'] = 1e-4
    if cfg['save_dir'] == '': cfg['save_dir'] = 'save/lr%s_a%s_b%s_e%i/' % (str(cfg['lr']), str(cfg['dv_prob']), str(cfg['dv_weight']), cfg['max_epochs'])
    if not os.path.exists(cfg['save_dir']): os.mkdir(cfg['save_dir'])
    cfg = SimpleNamespace(**cfg)
    train_model(cfg)