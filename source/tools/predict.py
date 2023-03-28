
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *
from data import ECGDataset
from nets.nets import LightX3ECG
from engines import predict
import configVars

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
parser.add_argument("--multilabel", action = "store_true")
parser.add_argument("--num_gpus", type = int, default = 1)
args = parser.parse_args()
config = {
    "ecg_leads":[
        0, 1, 
        6, 
    ], 
    "ecg_length":5000, 

    "is_multilabel":args.multilabel, 
    "device_ids":list(range(args.num_gpus)), 
}

train_loaders = {
    "pred":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = f"{configVars.pathCasos}{args.dataset}/pred.csv", data_path = f"{configVars.pathCasos}{args.dataset}/CasosNumpy", 
            config = config, 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 64, 
        shuffle = False
    )
}

model = LightX3ECG(
    num_classes = args.num_classes, 
)

save_ckp_dir = f"{configVars.pathModelos}{args.dataset}"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)

predict(
    train_loaders, 
    model, 
    config,
    save_ckp_dir = save_ckp_dir, 
    training_verbose = True, 
)