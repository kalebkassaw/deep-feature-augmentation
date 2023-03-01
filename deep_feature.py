import torch
from model import model, backbone
import os
import torchvision.transforms as transforms
from data import read_image, occlude, torchcvt
import data
import cv2

device = 'cuda:5'
model = model.to(device)

'''
@torch.no_grad()
def backbone(x):
    hk = model.avgpool.register_forward_hook(get_activation('deep_feat'))
    out = model(x)
    act = activation['deep_feat'][:,:,0,0]
    hk.remove()
    return act
'''

occluders = os.listdir('occluder_images')
for z, occ in enumerate(occluders):
    i = read_image('/scratch/ovis-cls/images/0/train/cat/337_00003_fa95a6a2_img_0000004.jpg', torchcvt=False)

    o = torch.os.path.join('occluder_images', occ)
    if o.endswith('.png'): o = read_image(o, torchcvt=False)
    else: continue
    
    occ_im = occlude(i, o)#, 'center')
    cv2.imwrite('test_images/occ_im_%i.jpg' % z, occ_im)
    ud = torch.unsqueeze(torchcvt(occ_im), dim=0).to(device)
    i = torch.unsqueeze(data.val_pipeline(torchcvt(i)), dim=0).to(device)

    with torch.no_grad():
        occ_feat = backbone(ud)
        unocc_feat = backbone(i)
        df = occ_feat - unocc_feat
    
    torch.save(df, 'deep_features/df%i.pt' % z)