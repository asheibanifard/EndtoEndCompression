from utils import SIREN, Composite
from utils import make_coord, create_json_file, LR_image_producer
from model import SRNO3D
from data import vol_patch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('grayscale')

DEVICE = torch.device('cuda:0')
NUM_EPOCH = 10000

P_ = []
L_ = []
CR_ = []
BPP_ = []
SC_ = []
SR_PORTION_ = []
ch = 4
n_resblocks = 2
for channels in [4, 8, 16]:
    P = []
    L = []
    CR = []
    BPP = []
    SC = []
    SR_PORTION = []
    for SCALE in [2, 4, 8]:
        hr = vol_patch
        HR = torch.tensor(hr).unsqueeze(0).unsqueeze(0).to(DEVICE)
        LR = torch.tensor(LR_image_producer(hr, SCALE)).unsqueeze(
            0).unsqueeze(0).to(DEVICE)
        hr_coord = make_coord([64, 64, 64], flatten=False).to(DEVICE)
        idx = torch.tensor(np.random.choice(
            HR.shape[-1]**3, LR.shape[-1]**3, replace=False))
        hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
        hr_coord = hr_coord.view(HR.shape[-3],
                                 HR.shape[-2],
                                 HR.shape[-1],
                                 hr_coord.shape[-1]).to(DEVICE)
        inp = LR
        cell = torch.tensor([[2 / HR.shape[-3], 2 / HR.shape[-2],
                              2 / HR.shape[-1]]],
                            dtype=torch.float32).permute(-1, -2).to(DEVICE)

        # Calculate bpp and cr
        NAME = 'SRNO3D'
        srno3d = SRNO3D(
            SCALE,
            hr_coord,
            cell,
            channels,
            n_resblocks,
            width=4,
            blocks=2).to(DEVICE)
        siren = SIREN(3, 32, 0, [4]).to(DEVICE)
        model = Composite(siren, srno3d).to(DEVICE)
        srno3d_NUM_PAR = sum(p.numel() for p in srno3d.parameters())
        NUM_PAR = sum(p.numel() for p in model.parameters())
        SR_PORTION.append((srno3d_NUM_PAR / NUM_PAR) * 100)
        bpp = (NUM_PAR * 4 * 8) / (64 * 64 * 64)
        cr = (1 - ((NUM_PAR) / (64 * 64 * 64))) * 100
        PSNR = []
        LOSS = []
        EPOCH = []
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=15e-4)
        INIT_LOSS = 1.0
        for epoch in range(NUM_EPOCH):
            optimizer.zero_grad()
            img = model(inp)
            loss = criterion(img, HR)
            if loss.item() < INIT_LOSS:
                best_loss = loss.item()
                INIT_LOSS = best_loss
                SR = model(inp)
                torch.save(
                    model.state_dict(),
                    f'models/model_scale__channel{channels}_\
                    scale{SCALE}_name{NAME}.pth')
            LOSS.append(best_loss)
            loss.backward()
            optimizer.step()
            best_psnr = -10. * np.log10(best_loss)
            PSNR.append(best_psnr)
            EPOCH.append(epoch)
            print(
                f'Scale: {SCALE}, resblocks: {n_resblocks},\
                {epoch}/{NUM_EPOCH}, best_psnr: {best_psnr},\
                cr: {cr}, bpp: {bpp}')
        P.append(best_psnr)
        L.append(best_loss)
        SC.append(SCALE)
        BPP.append(bpp)
        CR.append(cr)
        data = []
        data = dict(data)
        data['scale'] = SCALE
        data['Num_epochs'] = NUM_EPOCH
        data['Parameters'] = NUM_PAR
        data['bst_psnr'] = best_psnr
        data['bst_loss'] = best_loss
        create_json_file(data)
        plt.figure(figsize=(15, 15))
        plt.subplot(221)
        plt.title('PSNR (db)')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR(db)')
        plt.grid(True)
        plt.plot(EPOCH, PSNR)
        plt.subplot(222)
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.plot(EPOCH, LOSS)
        plt.subplot(223)
        plt.title('Original Image')
        plt.imshow(HR.squeeze(0).squeeze(0)[63].cpu().detach().numpy())
        plt.subplot(224)
        plt.title('Reconstructed Image')
        plt.imshow(SR.squeeze(0).squeeze(0)[63].cpu().detach().numpy())
        plt.savefig(
            f'./experiments/fig_scale{SCALE}_resblock_\
            {n_resblocks}_name{NAME}.pdf')
        plt.close()
    P_.append(P)
    L_.append(L)
    CR_.append(CR)
    BPP_.append(BPP)
    SC_.append(SC)
    SR_PORTION_.append(SR_PORTION)
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('PSNR(db) - Scale')
    plt.xlabel('Scale')
    plt.ylabel('PSNR(db)')
    plt.grid(True)
    plt.plot(SC, P, '-*')
    plt.subplot(222)
    plt.xlabel('BPP')
    plt.ylabel('PSNR(db)')
    plt.title('PSNR(db) - Bit Per Pixel(BPP)')
    plt.grid(True)
    plt.plot(BPP, P, '-*')
    plt.subplot(223)
    plt.xlabel('Compression Rate(%)')
    plt.ylabel('PSNR(db)')
    plt.title('PSNR(db) - Compression Rate(%)')
    plt.grid(True)
    plt.plot(CR, P, '-*')
    plt.subplot(224)
    plt.xlabel('Compression Rate(%)')
    plt.ylabel('PSNR(db)')
    plt.title('Compression Rate(%) - SR model occupation Rate(%)')
    plt.grid(True)
    plt.plot(CR, SR_PORTION, '-*')
    plt.savefig(f'./experiments/name{NAME}_channels{channels}.pdf')
    plt.close()
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('PSNR(db) - Scale')
plt.xlabel('Scale')
plt.ylabel('PSNR(db)')
plt.grid(True)
for i in range(len(P_)):
    plt.plot(SC_[i], P_[i], '-*')
plt.subplot(222)
plt.xlabel('BPP')
plt.ylabel('PSNR(db)')
plt.title('PSNR(db) - Bit Per Pixel(BPP)')
plt.grid(True)
for i in range(len(P_)):
    plt.plot(BPP_[i], P_[i], '-*')
plt.subplot(223)
plt.xlabel('Compression Rate(%)')
plt.ylabel('PSNR(db)')
plt.title('PSNR(db) - Compression Rate(%)')
plt.grid(True)
for i in range(len(P_)):
    plt.plot(CR_[i], P_[i], '-*')
plt.subplot(224)
plt.xlabel('Compression Rate(%)')
plt.ylabel('PSNR(db)')
plt.title('Compression Rate(%) - SR model occupation Rate(%)')
plt.grid(True)
for i in range(len(P_)):
    plt.plot(CR_[i], SR_PORTION_[i], '-*')
plt.savefig(f'./experiments/Total_plot_{NAME}.pdf')
