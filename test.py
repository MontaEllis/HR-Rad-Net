import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
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


mode = 'test'

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




source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir
rad_test = get_pyradiomics(hp.rad_test)
label_test_file = hp.label_test_file
images_test_indexs, label_test_indexs = label_check(hp.csv_test_path)





def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default='0306_all', required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=500000, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=1, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=1, help='batch-size')  
    training.add_argument('--sample', type=int, default=4, help='number of samples during training')  

    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=True,
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

def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Classification Testing')
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


    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])

    else:
        elapsed_epochs = 0

    model.cuda()



    test_dataset = MedData_test(source_test_dir,label_test_dir)
    test_loader = DataLoader(test_dataset.testing_set, 
                            batch_size=args.batch, 
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)


    model.eval()

    labelss = []
    gtss = []
    pdd = pd.DataFrame()
    for i, batch in enumerate(test_loader):
        


        x = batch['source']['data']
        y = batch['label']['data']

        img_index = batch['source']['path']
        gt = []
        rad = []
        for z in img_index:


            index = images_test_indexs.index(z.split('/')[-2])
            gt.append(int(label_test_indexs[index]))
            rad.append(np.array(rad_test[index],dtype=np.float32)[np.newaxis, :])         

        
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

        xxx = labels.clone()
        labels[labels>0.5] = 1
        labels[labels<=0.5] = 0






        labelss.append(labels.cpu().detach().numpy())
        gtss.append(gt.cpu().detach().numpy())



    acc = metrics.accuracy_score(np.concatenate(labelss).flatten(), np.concatenate(gtss).flatten()) 
    recall = metrics.recall_score(np.concatenate(labelss).flatten(), np.concatenate(gtss).flatten())
    f1 = metrics.f1_score(np.concatenate(labelss).flatten(), np.concatenate(gtss).flatten())

    print(acc)
    print(recall)
    print(f1)

      



if __name__ == '__main__':
    test()
   