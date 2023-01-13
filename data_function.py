from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
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
from pathlib import Path

from hparam import hparams as hp


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        self.subjects = []


        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob('*/IMG*.nii'))
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob('*/U*.nii'))



        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)



        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        # one_subject = self.training_set[0]
        # one_subject.plot()

    def transform(self):

        training_transform = Compose([
        # ToCanonical(),
        RandomMotion(),
        RandomBiasField(),
        ZNormalization(),
        RandomNoise(),
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),])


        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        self.subjects = []



        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob('*/*_source.nii'), key=lambda k: int(str(k).split('/')[-2].replace('(','').replace(')','')))
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob('*/*_label.nii'), key=lambda k: int(str(k).split('/')[-2].replace('(','').replace(')','')))


        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)



        self.transforms = self.transform()

        self.testing_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)



    def transform(self):

        testing_transform = Compose([
        ZNormalization(),
        ])


        return testing_transform