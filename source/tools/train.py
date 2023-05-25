
import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *
from data import ECGDataset
from nets.nets import LightX3ECG
from engines import train_fn
import configVars

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str), parser.add_argument("--num_classes", type = int)
parser.add_argument("--multilabel", action = "store_true")
parser.add_argument("--num_gpus", type = int, default = 1)
args = parser.parse_args()
print(args)
config = {
    "ecg_leads":[
        0, 1, 
        6, 
    ], 
    "ecg_length":4096, 

    "is_multilabel":args.multilabel, 
    "device_ids":list(range(args.num_gpus)), 
}

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = f"{configVars.pathCasos}{args.dataset}/train.csv", data_path = f"../../Examenes_Antonio_NPY/", 
            # df_path = f"{configVars.pathCasos}{args.dataset}/train.csv", data_path = f"{configVars.pathCasos}{args.dataset}/CasosNumpy", 
            config = config, 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 100, #63
        shuffle = True
        ,drop_last=True
    ), 
    "val":torch.utils.data.DataLoader(
        ECGDataset(
            df_path = f"{configVars.pathCasos}{args.dataset}/val.csv", data_path = f"../../Examenes_Antonio_NPY/", 
            # df_path = f"{configVars.pathCasos}{args.dataset}/val.csv", data_path = f"{configVars.pathCasos}{args.dataset}/CasosNumpy", 
            config = config, 
            augment = False, 
        ), 
        num_workers = 8, batch_size = 100, #56
        shuffle = False
        ,drop_last=True
    ), 
}
model = LightX3ECG(
    num_classes = args.num_classes, 
)
if not config["is_multilabel"]:
    criterion = F.cross_entropy
else:
    criterion = F.binary_cross_entropy_with_logits
optimizer = optim.Adam(
    model.parameters(), 
    lr = 1e-3, weight_decay = 5e-5, 
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    eta_min = 1e-4, T_max = 40, 
)

# save_ckp_dir = "../ckps/{}/{}".format(args.dataset, "LightX3ECG")
save_ckp_dir = f"{configVars.pathModelos}{args.dataset}"
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)

#Consultar los contenidos de los train_loarders que dan error.
# print("####################################################################")
# print("####################################################################")
# print(train_loaders)
# print(train_loaders["train"])
# print(train_loaders["val"])
# print(next(iter(train_loaders["train"])))
# print(next(iter(train_loaders["val"])))
# print("####################################################################")
# print("####################################################################")

train_fn(
    args.dataset,
    train_loaders, 
    model, 
    num_epochs = 5, 
    config = config, 
    criterion = criterion, 
    optimizer = optimizer, 
    scheduler = scheduler,
    save_ckp_dir = save_ckp_dir, 
)