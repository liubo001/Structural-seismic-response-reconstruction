import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn.functional import leaky_relu

next2pow = lambda x: int(2 ** torch.ceil(torch.log2(torch.tensor(x, dtype=torch.float32))))

def solve_sdof_eqwave_freq(omg, zeta, ag, dt):
    n = ag.shape[1]
    Nfft = next2pow(n) * 2
    ag_copy = ag.clone().detach().to(dtype=torch.float32, device=device)
    af = torch.fft.fft(ag_copy, Nfft)  # FFT for batch
    f = torch.fft.fftfreq(Nfft, d=dt, device=device)  # Frequency points
    Omg = 2.0 * torch.pi * f  # Circular frequency points
    H_u = -1.0 / (omg**2 - Omg**2 + 2.0 * zeta * omg * Omg * 1.0j)
    u = torch.fft.ifft(af * H_u, Nfft).real[:, :n]
    v = torch.fft.ifft(af * Omg * H_u, Nfft).real[:, :n]

    return u, v

def response_spectra(ag, dt, T, zeta):
    N = len(T)
    batch_size = ag.shape[0]
    RSA = torch.zeros(batch_size, N, dtype=torch.float32, device=ag.device)

    for i in range(N):
        omg = 2.0 * torch.pi / T[i]
        u, v = solve_sdof_eqwave_freq(omg, zeta, ag, dt)
        a = -2.0 * zeta * omg * v - omg * omg * u
        RSA[:, i] = torch.amax(torch.abs(a), dim=1)

    return RSA

def modify_mse(y_true, y_pred):
    y_true0 = y_true.squeeze(1)
    y_pred0 = y_pred.squeeze(1)

    y_true_max = torch.abs(y_true0).max(dim=1, keepdim=True).values
    y_true = y_true0 / y_true_max
    y_pred = y_pred0 / y_true_max
    err1 = torch.mean((y_true - y_pred) ** 2, dim=1)

    T = np.linspace(0.1, 4, 40)
    RSA_true0 = response_spectra(y_true, 0.02, T, zeta=0.05)
    RSA_pred0 = response_spectra(y_pred, 0.02, T, zeta=0.05)

    RSA_true_max = RSA_true0.max(dim=1, keepdim=True).values
    RSA_true = RSA_true0 / RSA_true_max
    RSA_pred = RSA_pred0 / RSA_true_max
    err2 = torch.mean((RSA_true - RSA_pred) ** 2, dim=1)

    mmse_value = torch.mean(err1 + err2)

    return mmse_value

def normalize(values, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in values]

def get_dataset(dataname):
    GM_data = pd.read_excel("./GMNames.xlsx")
    GM_data_li = GM_data.values.tolist()
    AsList = []
    for s_li in GM_data_li:
        AsList.append(s_li[0])

    StrInfos = []
    GMs = []
    TopAccs = []
    story_onehots = []
    floorAccs = []

    for i in range(len(dataname)):
    # for i in range(3):
        filename = dataname[i]
        GMNum = str(filename).split("-")[0]

        intensity = int(str(filename).split("-")[1]) - 6
        story_num = int(str(filename).split("-")[2])
        LCol = int(float(str(filename).split("-")[4]))
        LSpan = int(float(str(filename).split("-")[-3]))
        LBeam1 = int(float(str(filename).split("-")[-2]))
        LBeam2 = int(float(str(filename).split("-")[-1]))

        intensity_tensor = torch.tensor(intensity, dtype=torch.long)
        intensity_onehot = torch.nn.functional.one_hot(intensity_tensor, num_classes=3).float()
        # print(intensity_onehot)

        story_num_minMax = normalize(np.array(story_num).reshape(-1, 1), 3, 12)
        LCol_minMax = normalize(np.array(LCol).reshape(-1, 1), 3000, 3600)
        LSpan_minMax = normalize(np.array(LSpan).reshape(-1, 1), 3600, 7200)
        LBeam1_minMax = normalize(np.array(LBeam1).reshape(-1, 1), 5400, 6600)
        LBeam2_minMax = normalize(np.array(LBeam2).reshape(-1, 1), 2400, 3600)

        As_path = AsList[int(GMNum)]
        GM0 = np.loadtxt("./GM/" + As_path, dtype=float)*9.80
        FloorResponses0 = np.loadtxt('./NAcc/' + filename + '.txt', dtype=float)/100

        GM0 = GM0/np.max(np.abs(GM0))
        FloorResponses1 = FloorResponses0/np.max(np.abs(GM0))
        FloorResponses = FloorResponses1[0:3000,1:]

        GM = np.vstack([GM0.copy() for _ in range(FloorResponses.shape[1]-1)])
        TopAcc0 = FloorResponses[:, -1]
        TopAcc = np.vstack([TopAcc0.copy() for _ in range(FloorResponses.shape[1]-1)])
        floorAcc = np.vstack([FloorResponses[:,i] for i in range(FloorResponses.shape[1]-1)])

        story = np.hstack([np.full(1, i) for i in range(FloorResponses.shape[1]-1)])
        story_tensor = torch.tensor(story, dtype=torch.long)
        story_onehot = torch.nn.functional.one_hot(story_tensor, num_classes=12).float()

        intensity_onehots = np.array([intensity_onehot.numpy().copy() for _ in range(story.shape[0])])
        story_num_minMaxs = np.tile(story_num_minMax, story.shape[0]).reshape(-1, 1)
        LCol_minMaxs = np.tile(LCol_minMax, story.shape[0]).reshape(-1, 1)
        LSpan_minMaxs = np.tile(LSpan_minMax, story.shape[0]).reshape(-1, 1)
        LBeam1_minMaxs = np.tile(LBeam1_minMax, story.shape[0]).reshape(-1, 1)
        LBeam2_minMaxs = np.tile(LBeam2_minMax, story.shape[0]).reshape(-1, 1)
        StrInfo = np.column_stack((intensity_onehots, story_num_minMaxs, LCol_minMaxs, LSpan_minMaxs, LBeam1_minMaxs, LBeam2_minMaxs))

        StrInfos.extend(StrInfo)
        GMs.append(GM)
        TopAccs.append(TopAcc)
        story_onehots.append(story_onehot)
        floorAccs.append(floorAcc)

    StrInfos = np.array(StrInfos)[:, 3:4]#(8,1)
    stories = np.concatenate(story_onehots)#(12,1)
    GMs = np.concatenate(GMs)
    TopAccs = np.concatenate(TopAccs)
    floorAccs = np.concatenate(floorAccs)

    Str = np.hstack([StrInfos, stories])#(13,1)

    X = [Str, GMs, TopAccs]
    Y = floorAccs

    print('Shape of dataset X : {}'.format(GMs.shape))
    print('Shape of dataset Y : {}'.format(Y.shape))

    return X, Y

class Dataset:
    def __init__(self, filepath):
        self.train_path = filepath
    def load(self):
        data0 = pd.read_excel(self.train_path)
        columnsnames = data0.columns.values.tolist()
        dataname = data0[columnsnames[0]].values
        print(len(dataname))

        X_train_Mpaths, X_temp = train_test_split(dataname, test_size=0.4, random_state=0)
        X_val_Mpaths, X_test_Mpaths = train_test_split(X_temp, test_size=0.5, random_state=0)

        X_train, Y_train = get_dataset(X_train_Mpaths)
        X_val, Y_val = get_dataset(X_val_Mpaths)

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

class GMModel(nn.Module):
    def __init__(self):
        super(GMModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),  # 'same' padding
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # pooling
            nn.LeakyReLU(0.05),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=3, padding=0),  # pooling
            nn.LeakyReLU(0.05),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # pooling
            nn.LeakyReLU(0.05),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),  # pooling
            nn.LeakyReLU(0.05),
        )

    def forward(self, x):
        return self.model(x)

class StrModel(nn.Module):
    def __init__(self):
        super(StrModel, self).__init__()
        self.dense = nn.Linear(13, 16)
        self.layers = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(0.05),
            nn.Conv1d(1, 8, kernel_size=2, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(8, 16, kernel_size=2, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(32, 64, kernel_size=4, padding='valid'),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        x = leaky_relu(self.dense(x), 0.05)
        x = x.unsqueeze(1)
        return self.layers(x)

class DecoderModel(nn.Module):
    def __init__(self):
        super(DecoderModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(128, 125, kernel_size=5, padding=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(125, 64, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=3),
            nn.LeakyReLU(0.05),
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(0.05),
            nn.Conv1d(8, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.layers(x)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Model_StrInfo_NN(nn.Module):
    def __init__(self):
        super(Model_StrInfo_NN, self).__init__()
        self.GM_model = GMModel()
        self.str_model = StrModel()
        self.decoder_model = DecoderModel()
        self.conv1d = nn.Conv1d(128, 64, kernel_size=5, padding=2)

    def forward(self, str_input, GM_input, TopAcc_input):
        str_out = self.str_model(str_input)
        GM_out = self.GM_model(GM_input)
        TopAcc_out = self.GM_model(TopAcc_input)
        out111 = torch.cat([TopAcc_out, GM_out], dim=1)
        out111 = self.conv1d(out111)
        concatenated_input = torch.cat([str_out, out111], dim=1)
        return self.decoder_model(concatenated_input)

    def train_epoch(self, train_loader, optimizer, criterion, device):
        running_loss = 0.0
        for batch_idx, (str_input, GM_input, TopAcc_input, targets) in enumerate(train_loader):
            GM_input = GM_input.unsqueeze(1)
            TopAcc_input = TopAcc_input.unsqueeze(1)
            targets = targets.unsqueeze(1)

            str_input, GM_input, TopAcc_input, targets = (
                str_input.to(device),
                GM_input.to(device),
                TopAcc_input.to(device),
                targets.to(device)
            )
            optimizer.zero_grad()
            outputs = self(str_input, GM_input, TopAcc_input)
            # loss = criterion(targets, outputs)
            loss = modify_mse(targets, outputs)
            # print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    def val_epoch(self, val_loader, criterion, device):
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (str_input, GM_input, TopAcc_input, targets) in enumerate(val_loader):
                GM_input = GM_input.unsqueeze(1)
                TopAcc_input = TopAcc_input.unsqueeze(1)
                targets = targets.unsqueeze(1)

                str_input, GM_input, TopAcc_input, targets = (
                    str_input.to(device),
                    GM_input.to(device),
                    TopAcc_input.to(device),
                    targets.to(device)
                )
                
                outputs = self(str_input, GM_input, TopAcc_input)
                # loss = criterion(targets, outputs)
                loss = modify_mse(targets, outputs)
                # print(f"Validation Batch {batch_idx + 1}/{len(val_loader)}, Loss: {loss.item():.4f}")
                val_loss += loss.item()
                
        return val_loss / len(val_loader)

    def train_model(self, dataset, batch_size, nb_epoch):
        self.to(device)

        X_train = [torch.tensor(x, dtype=torch.float32).to(device) for x in dataset.X_train]
        Y_train = torch.tensor(dataset.Y_train, dtype=torch.float32).to(device)
        X_val = [torch.tensor(x, dtype=torch.float32).to(device) for x in dataset.X_val]
        Y_val = torch.tensor(dataset.Y_val, dtype=torch.float32).to(device)

        str_input_train, GM_input_train, TopAcc_input_train = X_train
        str_input_val, GM_input_val, TopAcc_input_val = X_val

        train_loader = DataLoader(
            TensorDataset(str_input_train, GM_input_train, TopAcc_input_train, Y_train),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(str_input_val, GM_input_val, TopAcc_input_val, Y_val),
            batch_size=batch_size, shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        train_losses = []
        val_losses = []
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
        for epoch in range(nb_epoch):
            self.train()
            running_loss = self.train_epoch(train_loader, optimizer, criterion, device)
            train_losses.append(running_loss)

            val_loss = self.val_epoch(val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Epoch [{epoch + 1}/{nb_epoch}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")

            torch.save(self.state_dict(), f"./model/model_{epoch + 1}_{val_loss:.4f}.pth")
            # print(f'Model saved at epoch {epoch + 1}')

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        np.savetxt("./result_data/train_losses.txt", train_losses, fmt="%s")
        np.savetxt("./result_data/val_losses.txt", val_losses, fmt="%s")
        print("Training and validation losses saved.")

        iters = range(epoch + 1)
        plt.figure()
        plt.plot(iters, train_losses, 'k', label='train loss')
        plt.plot(iters, val_losses, 'r', label='val loss')
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig("./result_data/loss.png")

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./result_data"):
        os.mkdir("./result_data")
    if not os.path.exists("./model"):
        os.mkdir("./model")
    
    filepath = "./TH_info.xlsx"
    batch_size, nb_epoch = 256, 100

    dataset = Dataset(filepath=filepath)
    dataset.load()
    model = Model_StrInfo_NN()
    start = time.time()
    model.train_model(dataset, batch_size, nb_epoch)
    end = time.time()
    print("总耗时=")
    print(end-start)
