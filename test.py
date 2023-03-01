import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import model as m
from data import get_dataloader, train_pipeline, val_pipeline, dv, ovis_categories
from tqdm import tqdm
import sklearn.metrics as skm
from types import SimpleNamespace
import os
import json

def qprint(x): 
    if not cfg.quiet: print(x)

def test_model(cfg):
    model = torch.load(cfg.checkpoint, map_location=cfg.device)
    val_accs = [0., 0., 0.]
    val_datas = get_dataloader('val', list(range(3)), batch_size=512, shuffle=False)#, list(range(3)), itrans=val_pipeline, batch_size=64)
    pbar = tqdm(range(3))
    model.eval()
    class_scores_all = [[],[],[]]
    preds_all = [[],[],[]]
    lbls_all = [[],[],[]]
    for olidx in pbar:
        val_data = val_datas[olidx]
        for bidx, (img, lbl) in enumerate(val_data):
            img = img.to(cfg.device)
            pbar.set_description(desc='Val%i batch %i/%i, acc: (0) %.3f (1) %.3f (2) %.3f' % (olidx, bidx, len(val_data), val_accs[0], val_accs[1], val_accs[2]))
            with torch.no_grad():
                class_scores = model(img).detach().cpu().numpy()
                class_scores_all[olidx].append(class_scores) 
                pred = np.argmax(class_scores, axis=1).astype(int)
                preds_all[olidx].append(pred)
                lbl = np.array([x.item() for x in lbl]).astype(int)
                lbls_all[olidx].append(lbl)
        class_scores_all[olidx] = np.concatenate(class_scores_all[olidx], axis=0)
        preds_all[olidx] = np.concatenate(preds_all[olidx], axis=0)
        lbls_all[olidx] = np.concatenate(lbls_all[olidx], axis=0)
        val_accs[olidx] = skm.accuracy_score(np.ravel(preds_all[olidx]), np.ravel(lbls_all[olidx]))
    class_scores_all = np.concatenate(class_scores_all, axis=0).tolist()
    preds_all = np.concatenate(preds_all, axis=0).tolist()
    lbls_all = np.concatenate(lbls_all, axis=0).tolist()
    prednames_all = [ovis_categories[p] for p in preds_all]

    pkg = {"class_scores": class_scores_all,
        "pred_label": preds_all,
        "pred_class": prednames_all}
    
    with open(cfg.save_dir, 'w') as wf:
        json.dump(pkg, wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='model save checkpoint (relative path OK)')
    parser.add_argument('--device', help='device to use', type=str, default='5')
    parser.add_argument('--save-dir', help='save directory for results', type=str, default='')
    parser.add_argument('--max-epochs', help='maximum number of epochs', type=int, default=20)
    parser.add_argument('--quiet', help='suppress intermediate details', action='store_true')
    cfg = vars(parser.parse_args())
    if cfg['save_dir'] == '':
        cfg['save_dir'] = cfg['checkpoint'].replace('.pt', '.json')
    print(cfg)
    cfg['device'] = 'cuda:%s' % cfg['device']
    cfg = SimpleNamespace(**cfg)
    test_model(cfg)