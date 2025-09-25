import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
# logging
import logging
from deeplens import Zernike
import matplotlib.pyplot as plt
import glob
import json
from scipy.optimize import lsq_linear

class resblock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(resblock, self).__init__()
        self.non_linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size)
        )
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        y1 = self.non_linear(x)
        y2 = self.linear(x)
        return y1 + y2
    
class MLP_ResNet(nn.Module):
    def __init__(self, input_size=63, output_size=15, hidden_size=128):
        super(MLP_ResNet, self).__init__()
        self.resblock1 = resblock(input_size*2, hidden_size, hidden_size)
        self.resblock2 = resblock(hidden_size, output_size, hidden_size//2)

    
    def forward(self, x):
        x = x
        x = torch.cat([x, x**2],dim=1)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x


class AE(nn.Module):
    """an autoencoder model, where the voltage is served as the latent space"""
    def __init__(self,in_size=15, latent_size=63,hidden_size=128):
        super(AE, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(in_size, hidden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_size, latent_size),
        #     nn.Sigmoid()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_size, hidden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_size, in_size)
        # )
        self.encoder = nn.Sequential(
            resblock(in_size, hidden_size, hidden_size),
            resblock(hidden_size, hidden_size, hidden_size),
            resblock(hidden_size, latent_size, hidden_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            resblock(latent_size*2, hidden_size, hidden_size),
            resblock(hidden_size, hidden_size, hidden_size),
            resblock(hidden_size, in_size, hidden_size)
        )
    def forward(self, x):
        latent = self.encoder(x)
        latent = (latent+1)/2.0 # scale latent to [0,1]
        x = torch.cat([latent, latent**2],dim=1)
        x = self.decoder(x)
        return latent,x

class Linear_decoder(nn.Module):
    def __init__(self, output_size, calib_file = "./site-packages/dpp_ctrl/calibrations/Calibration_D7_001B_C_AU.json"):
        super(Linear_decoder, self).__init__()

        with open(calib_file) as f:
            calib = json.load(f)
        self.infl_matrix = torch.tensor(calib['Vertical Influence Matrix']) * (270**2)
        self.flat_field = torch.tensor(calib['Vertical Flat Offsets'])
        print(f"infl_matrix.shape: {self.infl_matrix.shape}") # should be (91,63)
        print(f"flat_field.shape: {self.flat_field.shape}") # should be (91,)

        self.param = nn.Parameter(torch.zeros(output_size))
        self.scale = nn.Parameter(torch.ones(1))
        self.output_size = output_size
    
    def to(self,device):
        self.infl_matrix = self.infl_matrix.to(device)
        self.flat_field = self.flat_field.to(device)
        return super().to(device)

    def forward(self, volt):
        # multiply volt in (N,63) with infl_matrix (91,63) to get zernike coefficients (N,91)
        zern = torch.matmul(volt, self.infl_matrix.T) + self.flat_field
        zern = zern[...,:self.output_size]
        zern =(zern + self.param) * self.scale
        return zern
    
    def inverse(self, zern):
        zern = zern/self.scale - self.param
        # append 0 to zern to make it 91 dim
        zern = torch.cat([zern,torch.zeros(zern.shape[0],91-self.output_size).to(zern.device)],dim=1)
        zern = zern - self.flat_field
        # inverse the zernike coefficients to voltages, volt in [0,1]
        A = self.infl_matrix.cpu().double()
        volts = []
        for i in range(len(zern)):
            voltages = lsq_linear(A=A, b=zern[i].cpu().double(), bounds=(0, 1))
            volts.append(np.sqrt(voltages.x))
        # voltages = lsq_linear(A=self.infl_matrix.cpu(), b=zern.cpu(), bounds=(0, 1))
        volts = np.stack(volts)
        volts = torch.tensor(volts).float().to(zern.device)
        return volts
    

class Decoder(nn.Module):
    def __init__(self, input_size=63, output_size=15, hidden_size=64):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc_linear = nn.Linear(input_size*2, output_size)
        self.non_linear = nn.Sigmoid()
        

    def forward(self, x):
        x = x # normalize x
        x = torch.cat([x, x**2],dim=1) # add x^2 term

        out = self.fc1(x)
        out = self.non_linear(out)
        out = self.fc2(out)
        
        y = self.fc_linear(x)

        out = out + y

        return out


class Encoder(nn.Module):
    def __init__(self, input_size=15, output_size=63, hidden_size=64,N_layers=2):
        super(Encoder, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 63)
        self.non_linear = nn.Sigmoid()
        # self.non_linear = nn.Hardsigmoid()

        # self.model = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     self.non_linear,
        #     nn.Linear(hidden_size, hidden_size),
        #     self.non_linear,
        #     nn.Linear(hidden_size, output_size),
        #     self.non_linear
        # )
        
        self.layers = nn.ModuleList()
        for i in range(N_layers):
            if i == 0: # input layer
                self.layers.append(nn.Linear(input_size, hidden_size))
            elif i == N_layers-1: # output layer
                self.layers.append(nn.Linear(hidden_size, output_size))
            else:   # hidden layers
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.non_linear)

        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        # out = self.fc1(x)
        # out = self.non_linear(out)
        # out = self.fc2(out)
        # out = self.non_linear(out)
        out = self.model(x)
        out = torch.sqrt(out)
        return out

class lens_dataset(Dataset):
    def __init__(self,data_root,N_zern):
        self.data_root = data_root
        self.lens_files = sorted(glob.glob(f"{data_root}/*.json"))
        self.N_zern = N_zern
    def __len__(self):
        return len(self.lens_files)

    def __getitem__(self, idx):
        with open(self.lens_files[idx],'r') as f:
            lens = json.load(f)
        zern = torch.tensor(lens['surfaces'][0]['zern_amp'])*1000 # in um
        zern = zern[:self.N_zern]
        return zern

class VoltageZernikeDataset(Dataset):
    def __init__(self, data_root, split='train',N_zern=15, noise_zern = 0, noise_volt = 0):
        self.data_root = data_root
        n_order = int((-3+np.sqrt(1+8*N_zern))/2)
        zern = Zernike(n=n_order)
        self.coefnorm = (1/zern.coefnorm)**2
        self.coefnorm[1:] = self.coefnorm[1:]/self.coefnorm[1:].sum()
        self.coefnorm[0]=0
        self.coefnorm = torch.sqrt(self.coefnorm)
        assert ((self.coefnorm**2).sum()-1)<1e-6, f"coefnorm sum: {(self.coefnorm**2).sum()}"

        self.noise_zern = noise_zern
        self.noise_volt = noise_volt
        
        self.idxs = np.loadtxt(f"{data_root}/{split}_idxs.txt").astype(int)
        self.N_zern = N_zern

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        # read volt from txt file and convert to tensor
        with open(f'{self.data_root}/volt/{idx:03d}.txt','r') as f:
            volt = f.readlines()
            volt = torch.tensor([float(v.split('\n')[0]) for v in volt])
            volt = volt/270.0 # normalize voltage to [0,1]
        # read zern from txt file and convert to tensor
        with open(f'{self.data_root}/rec/{idx:03d}.txt','r') as f:
            zern = f.readlines()
            zern = torch.tensor([float(v.split('\n')[0]) for v in zern]) *1000 # in um
            zern_rec = zern[:self.N_zern]
        
        # read zern_dev from txt file and convert to tensor
        with open(f'{self.data_root}/dev/{idx:03d}.txt','r') as f:
            zern = f.readlines()
            zern = torch.tensor([float(v.split('\n')[0]) for v in zern]) *1000
            zern_dev = zern[:self.N_zern]

        # add noise to zern_rec
        noise = torch.randn_like(zern_rec)*self.noise_zern*self.coefnorm
        zern_rec = zern_rec + noise

        # add noise to volt
        noise = torch.randn_like(volt)*self.noise_volt
        volt = volt + noise
        volt = volt.clamp(0,1)

        return volt, zern_rec, zern_dev


# Test loop
def test_AE(encoder,decoder, test_loader, criterion,device='cuda'):
    encoder.eval()
    decoder.eval()

    test_loop = 0.0
    test_volt = 0.0
    
    with torch.no_grad():
        for voltages, zern_rec, _ in test_loader:
            voltages, zern_rec = voltages.to(device), zern_rec.to(device)

            volt_pred = encoder(zern_rec)
            loss_volt = criterion(volt_pred, voltages)

            # Forward pass using the pre-trained decoder
            zern_pred = decoder(volt_pred)
            loss_loop = criterion(zern_pred, zern_rec)
            
            test_loop += loss_loop.item()* len(voltages)
            test_volt += loss_volt.item()* len(voltages)

    test_loop /= len(test_loader.dataset)
    test_volt /= len(test_loader.dataset)
    print(f'Test Loss Loop: {test_loop:.4f}, Test Loss Volt: {test_volt:.4f}')
    logging.info(f'Test Loss Loop: {test_loop:.4f}, Test Loss Volt: {test_volt:.4f}')
    
    return test_loop, test_volt


def vis_volt(volt,title="Voltage (v)",fname='volt.png'):
    ## volt is a 63 dim tensor, visualize it as a 8x8 image
    # add zero to make it 64 dim
    volt = torch.cat([volt.cpu().detach(),torch.zeros(1)],dim=0)
    volt = volt.view(8,8)
    plt.imshow(volt)
    plt.title(title)
    plt.colorbar()
    plt.savefig(fname)
    plt.clf()

# Test loop
def test_decoder(model, test_loader, criterion, idx_bound=None,device='cuda', model_name='NN'):
    model.eval()
    test_loss = 0.0
    
    # print const shift parameter
    # print(linear_model.parameters())
    
    with torch.no_grad():
        for voltages, zern_rec, _ in test_loader:
            voltages, zern_rec = voltages.to(device), zern_rec.to(device)
            outputs = model(voltages)

            if idx_bound is not None:
                outputs = outputs[...,idx_bound[0]:idx_bound[1]]
                zern_rec = zern_rec[...,idx_bound[0]:idx_bound[1]]

            loss = criterion(outputs, zern_rec)
            test_loss += loss.item() * len(voltages)
        test_loss /= len(test_loader.dataset)
        print(f'Test Loss {model_name}: {test_loss:.4f}')
        logging.info(f'Test Loss {model_name}: {test_loss:.4f}')

    return test_loss  
    #     out = test_loss
    
    # if linear_model is not None:
    #     test_loss_dev = 0.0
    #     parameter = list(linear_model.parameters())
    #     with torch.no_grad():
    #         for voltages, zern_rec, zern_dev in test_loader:
    #         voltages, zern_rec, zern_dev = voltages.to(device), zern_rec.to(device), zern_dev.to(device)
    #         zern_dev = linear_model(voltages)

    #         if idx_bound is not None:
    #             zern_rec = zern_rec[...,idx_bound[0]:idx_bound[1]]
    #             zern_dev = zern_dev[...,idx_bound[0]:idx_bound[1]]

    #         loss_dev = criterion(zern_rec, zern_dev)
    #         test_loss_dev += loss_dev.item()* len(voltages)

    #     test_loss_dev /= len(test_loader.dataset)
    #     print(f'Test Loss Linear: {test_loss_dev:.4f}')
    #     logging.info(f'Test Loss Linear: {test_loss_dev:.4f}')
    
    #     out = test_loss, test_loss_dev
    
    # return out

# Test loop
def test_encoder(encoder, test_loader, criterion, device='cuda'):
    encoder.eval()
    test_loss = 0.0
    with torch.no_grad():
        for voltages, zern_rec, zern_dev in test_loader:
            voltages, zern_rec, zern_dev = voltages.to(device), zern_rec.to(device), zern_dev.to(device)
            volg_out = encoder(zern_rec)
            loss = criterion(volg_out, voltages)
            test_loss += loss.item() * len(voltages)

    test_loss /= len(test_loader.dataset)
    print(f'Test encoder Loss: {test_loss:.4f}')
    return test_loss


# Test each rmse error and zernike std
def test_std_rmse(model, linear_model, test_loader,idx_bound=None):
    model.eval()
    nn_rmse_list = []
    dev_rmse_list = []
    zern_std_list = []
    nn_scale_list = []
    dev_scale_list = []
    parameter = list(linear_model.parameters())
    with torch.no_grad():
        for voltages, zern_rec, zern_dev in test_loader:
            outputs = model(voltages)
            zern_dev = linear_model(voltages)

            if idx_bound is not None:
                outputs = outputs[...,idx_bound[0]:idx_bound[1]]
                zern_rec = zern_rec[...,idx_bound[0]:idx_bound[1]]
                zern_dev = zern_dev[...,idx_bound[0]:idx_bound[1]]
            
            loss = torch.norm(outputs-zern_rec,dim=1)
            loss_dev = torch.norm(zern_dev-zern_rec,dim=1)
            nn_rmse_list.append(loss)
            dev_rmse_list.append(loss_dev)
            zern_std_list.append(torch.norm(zern_rec,dim=1))
            nn_scale_list.append(torch.norm(outputs,dim=1)/torch.norm(zern_rec,dim=1))
            dev_scale_list.append(torch.norm(zern_dev,dim=1)/torch.norm(zern_rec,dim=1))
    nn_rmse = torch.cat(nn_rmse_list) # concatenate all the tensors (N,)
    dev_rmse = torch.cat(dev_rmse_list) # concatenate all the tensors (N,)
    zern_std = torch.cat(zern_std_list) # concatenate all the tensors (N,)
    nn_scale = torch.cat(nn_scale_list)
    dev_scale = torch.cat(dev_scale_list)

    return nn_rmse, dev_rmse, nn_scale, dev_scale, zern_std


def zernike_MSE(pred,gt):
    MSE = torch.sum((pred-gt)**2,dim=1).mean()
    return MSE

