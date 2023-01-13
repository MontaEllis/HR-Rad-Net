import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
import pandas as pd
import csv
from skimage import measure
import time
from sklearn import metrics
from sklearn.metrics import classification_report
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from medpy.io import load,save
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_pyradiomics(pyradiomics_dir):
    df = pd.read_csv(pyradiomics_dir)
    df = df.values
    df = df[:,24:]
    # cols=df.columns
    # print(df[0])
    return df

def label_check(csv_path):
    images_indexs = []
    label_indexs = []
    with open(csv_path,'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            images_indexs.append(line[0].split('/')[-2])
            label_indexs.append(line[2])
    return images_indexs, label_indexs

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir
rad_train = get_pyradiomics(hp.rad_train)
label_train_file = hp.lable_train_file
images_train_indexs, label_train_indexs = label_check(hp.csv_train_path)








def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default='0713_nose', required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=500000, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=2, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=6, help='batch-size')  
    training.add_argument('--sample', type=int, default=4, help='number of samples during training')  

    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--init-lr", type=float, default=0.0002, help="learning rate")

    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')


    return parser


transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((int(64),int(64))),
    transforms.ToTensor(),
])


transform_pil=transforms.Compose([
    transforms.ToPILImage(),
])

def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Classification Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


    from data_function import MedData_train, MedData_test
    os.makedirs(args.output_dir, exist_ok=True)


    from config import config
    from model import get_cls_net
    _config = config()
    model = get_cls_net(config=_config)




    model = torch.nn.DataParallel(model, device_ids=devicess,output_device=[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from loss_function import Classification_Loss
    criterion = Classification_Loss().cuda()


    writer = SummaryWriter(args.output_dir)



    train_dataset = MedData_train(source_train_dir,label_train_dir)
    train_loader = DataLoader(train_dataset.training_set, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)


    

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)


    

    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        epoch += elapsed_epochs

        train_epoch_avg_loss = 0.0
        num_iters = 0

        model.train()

        labelss = []
        gtss = []
        for i, batch in enumerate(train_loader):
            

            if hp.debug:
                if i >=1:
                    break

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()


            x = batch['source']['data']
            y = batch['label']['data']

            img_index = batch['source']['path']
            gt = []
            rad = []
            for z in img_index:
                index = images_train_indexs.index(z.split('/')[-2])

                gt.append(int(label_train_indexs[index]))
                rad.append(np.array(rad_train[index],dtype=np.float32)[np.newaxis, :])
            
            gt = torch.from_numpy(np.array(gt)).unsqueeze(1)
            rad = torch.from_numpy(np.concatenate(rad)).cuda()


            x = x.type(torch.FloatTensor).squeeze(4)#.repeat(1,3,1,1)
            y = y.squeeze(4).int()


            image_datas = []
            label_datas = []
            image_label_datas = []
            for j in range(y.shape[0]):

                properties = measure.regionprops(np.array(y[j].cpu()).astype(np.int8)[0])

                
                for pro in properties:
                    bbox = pro.bbox

                    image_data = transform(x[j][:,bbox[0]:bbox[2],bbox[1]:bbox[3]])
                    label_data = transform(y[j][:,bbox[0]:bbox[2],bbox[1]:bbox[3]])


                    image_label_data = torch.mul(image_data,label_data)

                    image_datas.append(image_data)
                    label_datas.append(label_data)
                    image_label_datas.append(image_label_data)

                    
                    break
            image_datas = torch.cat(image_datas).unsqueeze(1)
            label_datas = torch.cat(label_datas).unsqueeze(1)
            image_label_datas = torch.cat(image_label_datas).unsqueeze(1)

            inputt = torch.cat([image_datas,label_datas,image_label_datas],1).cuda()





            outputs = model(inputt,rad)


            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels>0.5] = 1
            labels[labels<=0.5] = 0


            loss = criterion(outputs, gt.float().cuda(),model)

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1

            print("loss:"+str(loss.item()))
            writer.add_scalar('Training/Loss', loss.item(),iteration)

            labelss.append(labels.cpu().detach().numpy())
            gtss.append(gt.cpu().detach().numpy())

        acc = metrics.precision_score(np.concatenate(labelss).flatten(), np.concatenate(gtss).flatten()) 
        recall = metrics.recall_score(np.concatenate(labelss).flatten(), np.concatenate(gtss).flatten())
        f1 = metrics.f1_score(np.concatenate(labelss).flatten(), np.concatenate(gtss).flatten())
        ## log
        writer.add_scalar('Training/acc', acc,epoch)
        writer.add_scalar('Training/recall', recall,epoch)
        writer.add_scalar('Training/f1', f1,epoch)

    

        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )




        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {
                    
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
        



    writer.close()



if __name__ == '__main__':
    train()
   