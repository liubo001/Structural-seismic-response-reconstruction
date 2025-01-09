import time
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import leaky_relu
from sklearn.model_selection import train_test_split

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
    GMpgas = []
    TopAccs = []
    story_onehots = []
    floorAccs = []

    for i in range(len(dataname)):
    # for i in range(10):
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

        story_num_minMax = normalize(np.array(story_num).reshape(-1, 1), 3, 12)
        LCol_minMax = normalize(np.array(LCol).reshape(-1, 1), 3000, 3600)
        LSpan_minMax = normalize(np.array(LSpan).reshape(-1, 1), 3600, 7200)
        LBeam1_minMax = normalize(np.array(LBeam1).reshape(-1, 1), 5400, 6600)
        LBeam2_minMax = normalize(np.array(LBeam2).reshape(-1, 1), 2400, 3600)

        As_path = AsList[int(GMNum)]
        GM0 = np.loadtxt("./GM/" + As_path, dtype=float)*9.80#unit:m/s^2
        FloorResponses0 = np.loadtxt('./NAcc/' + filename + '.txt', dtype=float)/100#unit:m/s^2

        GMpga = np.max(np.abs(GM0))
        GM0 = GM0 / GMpga
        FloorResponses0 = FloorResponses0 / GMpga
        FloorResponses = FloorResponses0[0:3000, 1:]#第一列为时间步

        GMpga1 = np.vstack([GMpga.copy() for _ in range(FloorResponses.shape[1] - 1)])
        GM = np.vstack([GM0.copy() for _ in range(FloorResponses.shape[1]-1)])
        TopAcc0 =  FloorResponses[:, -1]
        TopAcc = np.vstack([TopAcc0.copy() for _ in range(FloorResponses.shape[1]-1)])
        floorAcc = np.vstack([FloorResponses[:, i] for i in range(FloorResponses.shape[1]-1)])

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
        GMpgas.append(GMpga1)

    StrInfos = np.array(StrInfos)[:, 3:4]#(8,1)
    stories = np.concatenate(story_onehots)
    GMs = np.concatenate(GMs)
    TopAccs = np.concatenate(TopAccs)
    floorAccs = np.concatenate(floorAccs)
    GMpgas = np.concatenate(GMpgas)

    # print(GMpgas)
    # print(aaa)
    Str = np.hstack([StrInfos, stories])#(20,1)

    X = [Str, GMs, TopAccs]
    Y = floorAccs

    print('Shape of dataset X : {}'.format(GMs.shape))
    print('Shape of dataset Y : {}'.format(Y.shape))

    return X, Y, GMpgas

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

        # Save truth names
        with open("./predicts/truths_name.txt", "w") as f:
            truth_name_out = "\n".join(X_train_Mpaths)
            f.write(truth_name_out)

        X_test, Y_test, pga = get_dataset(X_test_Mpaths)
        np.savetxt("./predicts/truths_GMpga.txt", pga, fmt="%s")
        np.savetxt("./predicts/truths_name_story.txt", X_test[0], fmt="%s")
        np.savetxt("./predicts/truths.txt", Y_test, fmt="%s")

        self.X_test = X_test
        self.Y_test = Y_test

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

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def evaluate(self, dataset, batch_size):
        self.to(device)

        X_test = [torch.tensor(x, dtype=torch.float32).to(device) for x in dataset.X_test]
        Y_test = torch.tensor(dataset.Y_test, dtype=torch.float32).to(device)

        str_input_test, GM_input_test, TopAcc_input_test = X_test

        test_loader = DataLoader(
            TensorDataset(str_input_test, GM_input_test, TopAcc_input_test, Y_test),
            batch_size=batch_size, shuffle=False
        )

        all_predictions = []
        with torch.no_grad():
            for batch_idx, (str_input, GM_input, TopAcc_input, targets) in enumerate(test_loader):
                GM_input = GM_input.unsqueeze(1)
                TopAcc_input = TopAcc_input.unsqueeze(1)
                targets = targets.unsqueeze(1)

                str_input, GM_input, TopAcc_input, targets = (
                    str_input.to(device),
                    GM_input.to(device),
                    TopAcc_input.to(device),
                    targets.to(device)
                )
                
                predictions = self(str_input, GM_input, TopAcc_input)
                all_predictions.append(predictions.cpu().numpy())
                print(f"test Batch {batch_idx + 1}/{len(test_loader)}")

        predicts = np.concatenate(all_predictions, axis=1)
        predicts = predicts.squeeze()
        print(predicts.shape)
        np.savetxt("./predicts/predicts.txt", predicts, fmt="%s")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_path = "./predicts"
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    if not os.path.exists("./model"):
        print("请补充预训练模型")

    filepath = "./TH_info.xlsx"
    dataset = Dataset(filepath=filepath)
    dataset.load()

    model_name = "model_37_0.0071"
    batch_size = 1
    model_path = './model/%s.pth' % model_name

    model = Model_StrInfo_NN()
    model.load_model(model_path)

    start = time.time()
    model.evaluate(dataset, batch_size)

    end = time.time()
    print("Evaluation time:", end - start)