from os.path import exists, splitext, join, isdir, dirname
from os import listdir, mkdir, makedirs
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image
from ISR.models import RDN

rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/sample_weights/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5') #change the weight folder

input_folder = sys.argv[1]
output_folder = sys.argv[2]

def super_resolution(input_folder,output_folder):
    if not exists(output_folder):
        mkdir(output_folder)
    for i in get_image_list(input_folder):
        im_sr = sr(i)
        i = i.replace(input_folder.strip('/'),output_folder.strip('/'))
        folder = dirname(i)
        if not exists(folder):
            makedirs(folder)
        im_sr.save(splitext(i)[0]+'.jpeg')


def get_image_list(src_folder):
    image_list = []
    def go(folder_or_file_path):
        if isdir(folder_or_file_path):
            for i in listdir(folder_or_file_path):
                if i != 'PaxHeader':
                    i = join(folder_or_file_path, i)
                    i = go(i)
                    if i is not None and len(i):
                        image_list.append(i)
        else:
            if folder_or_file_path.endswith('.jpg'):
                return folder_or_file_path
    go(src_folder)
    return image_list

def sr(im_lr):
    img = Image.open(im_lr)
    lr_img = np.array(img)
    sr_img = rdn.predict(lr_img)
    img_new = Image.fromarray(sr_img)
    return img_new


tqdm(super_resolution(input_folder,output_folder))


