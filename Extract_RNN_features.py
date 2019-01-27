import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

#reading in argument number#
import sys
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))
selected_epoch=sys.argv[1]
print ('Selected Epoch:', selected_epoch)


# set path
action_name_path = './UCF101actions.pkl'
save_model_path = "./save_ckpt_2/"

# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
batch_size = 40
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1


with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(os.path.join(data_path, f))


# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# reset data loader
all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
all_data_loader = data.DataLoader(Dataset_CRNN(all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)


# reload CRNN model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNNOutput(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch'+str(selected_epoch)+'.pth')))
rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch'+str(selected_epoch)+'.pth')))
print('CRNN model reloaded!')


def CNN_features(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()

    feature_list = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)): #loader=all_data_loader
            # distribute data to device
            # print("iterating",batch_idx)
            X = X.to(device)
            batch_features = cnn_encoder(X)
            feature_list.extend(batch_features.cpu().numpy().tolist())
        return feature_list



def RNN_features(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()

    feature_list = []
    output_list = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)): #loader=all_data_loader
            # distribute data to device
            # print("iterating",batch_idx)
            X = X.to(device)
            batch_output, batch_features = rnn_decoder(cnn_encoder(X))       #careful: X, y are all given in batches of 40!
            feature_list.extend(batch_features.cpu().numpy().tolist())
            output_list.extend(batch_output.cpu().numpy().tolist())
        return feature_list, output_list  # 40*28*512, 40*101 for 1 batch


RNN_features, RNN_outputs= RNN_features([cnn_encoder, rnn_decoder], device, all_data_loader)


df = pd.DataFrame(data={'filename': fnames, 'y': cat2labels(le, all_y_list), 'features': RNN_features, 'Outputs': RNN_outputs}) #이걸로 하면 됨.
df.to_pickle("./UCF101_RNN_features"+str(selected_epoch)+"l.pth")  # save pandas dataframe
# To import this pickle datafile: 
# import pandas as pd
# import pickle
# a = pd.read_pickle("./UCF101_CNN_features.pkl")
print('RNN feature extraction finished!')
