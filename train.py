"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy.stats import entropy
from torch import nn
import datetime
import random


from trainer_council import Council_Trainer

from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, \
    load_inception

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys, traceback
import tensorboardX
import shutil
import threading
import torchvision.utils as vutils
import math
from scipy.stats import binom
from tqdm import tqdm
import time
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/glasses_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.outputs', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--cuda_device", type=str, default='cuda:0', help="gpu to run on")
opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
config['cuda_device'] = opts.cuda_device

# FOR REPRODUCIBILITY
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(config['random_seed'])

# Setup model and data loader
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

train_display_a_samples_idx = [np.random.randint(train_loader_a[0].__len__()) for _ in range(display_size)]
train_display_b_samples_idx = [np.random.randint(train_loader_b[0].__len__()) for _ in range(display_size)]
test_display_a_samples_idx = [np.random.randint(test_loader_a[0].__len__()) for _ in range(display_size)]
test_display_b_samples_idx = [np.random.randint(test_loader_b[0].__len__()) for _ in range(display_size)]


try:
    train_display_images_a = torch.stack([train_loader_a[0].dataset[i][0] for i in train_display_a_samples_idx]).cuda(config['cuda_device'])
    train_display_attrs_a = torch.tensor([train_loader_a[0].dataset[i][1] for i in train_display_a_samples_idx], dtype=torch.int).cuda(config['cuda_device'])
except:
    train_display_images_a = torch.stack([train_loader_a[0].dataset[i][0] for i in train_display_a_samples_idx]).cuda(config['cuda_device'])
    train_display_attrs_a = torch.tensor([train_loader_a[0].dataset[i][1] for i in train_display_a_samples_idx], dtype=torch.int).cuda(config['cuda_device'])
try:
    train_display_images_b = torch.stack([train_loader_b[0].dataset[i][0] for i in train_display_b_samples_idx]).cuda(config['cuda_device'])
    train_display_attrs_b = torch.tensor([train_loader_b[0].dataset[i][1] for i in train_display_b_samples_idx], dtype=torch.int).cuda(config['cuda_device'])
    
except:
    train_display_images_b = torch.stack([train_loader_b[0].dataset[i][0] for i in train_display_b_samples_idx]).cuda(config['cuda_device'])
    train_display_attrs_b = torch.tensor([train_loader_b[0].dataset[i][1] for i in train_display_b_samples_idx], dtype=torch.int).cuda(config['cuda_device'])
try:
    test_display_images_a = torch.stack([test_loader_a[0].dataset[i][0] for i in test_display_a_samples_idx]).cuda(config['cuda_device'])
    test_display_attrs_a = torch.tensor([test_loader_a[0].dataset[i][1] for i in test_display_a_samples_idx], dtype=torch.int).cuda(config['cuda_device'])
except:
    # test_display_images_a = torch.stack([test_loader_a[0].dataset[np.random.randint(test_loader_a[0].__len__())] for _ in range(display_size)]).cuda()
    test_display_images_a = None
try:
    test_display_images_b = torch.stack([test_loader_b[0].dataset[i][0] for i in test_display_b_samples_idx]).cuda(config['cuda_device'])
    test_display_attrs_b = torch.tensor([test_loader_b[0].dataset[i][1] for i in test_display_b_samples_idx], dtype=torch.int).cuda(config['cuda_device'])
except:
    test_display_images_b = torch.stack([test_loader_b[0].dataset[i][0] for i in test_display_b_samples_idx]).cuda(config['cuda_device'])
    test_display_attrs_b = torch.tensor([test_loader_b[0].dataset[i][1] for i in test_display_b_samples_idx], dtype=torch.int).cuda(config['cuda_device'])

trainer = Council_Trainer(config, config['cuda_device'])

trainer.cuda(config['cuda_device'])

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path, model_name)
checkpoint_directory, image_directory, log_directory = prepare_sub_folder(output_directory)

config_backup_folder = os.path.join(output_directory, 'config_backup')
if not os.path.exists(config_backup_folder):
    os.mkdir(config_backup_folder)
shutil.copy(opts.config, os.path.join(config_backup_folder, ('config_backup_' + str(datetime.datetime.now())[:19] + '.yaml').replace(' ', '_')))  # copy config file to output folder


m1_1_a2b, s1_1_a2b, m1_1_b2a, s1_1_b2a = None, None, None, None # save statisices for the fid calculation

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0


def launchTensorBoard(port=6006):
    import os
    os.system('tensorboard --logdir=' + log_directory + ' --port=' + str(port) + ' > /dev/null 2>/dev/null')
    return

if config['misc']['start_tensor_board']:
    port = config['misc']['start_tensor_board port']
    t_tensorBoard = threading.Thread(target=launchTensorBoard, args=([port]))  # launches TensorBoard in a diffrent thread
    t_tensorBoard.start()
    print(colored('tensorboard board launched at http://127.0.0.1:' + str(port), color='yellow', attrs=['underline', 'bold', 'blink', 'reverse']))
train_writer = tensorboardX.SummaryWriter(log_directory, purge_step=iterations)

def test_fid(dataset1, dataset2, iteration, train_writer, name, m1=None, s1=None, retun_m1_s1=False, batch_size=10, dims=2048, cuda=True):
    import pytorch_fid.fid_score
    fid_paths = [dataset1, dataset2]
    try:
        fid_value, m1, s1 = pytorch_fid.fid_score.calculate_fid_given_paths_save_first_domain_statistic(paths=fid_paths,
                                                                                                batch_size=batch_size,
                                                                                                cuda=cuda,
                                                                                                dims=dims,
                                                                                                m1=m1, s1=s1)

        train_writer.add_scalar('FID score/' + name, fid_value, iterations)

        print(colored('iteration: ' + str(iteration) + ' ,' + name + ' aprox FID: ' + str(fid_value), color='green', attrs=['underline', 'bold', 'blink', 'reverse']))

    except Exception as e:
        print(str(e))
        fid_value, m1, s1 = 0, None, None
    if not retun_m1_s1:
        return
    return m1, s1


t = time.time()
dis_iter = 1
try:
    while True:
        tmp_train_loader_a, tmp_train_loader_b = (train_loader_a[0], train_loader_b[0])
        for it, (images_a, images_b) in enumerate(zip(tmp_train_loader_a, tmp_train_loader_b)):

            attrs_a, attrs_b = images_a[1].cuda(config['cuda_device']).detach(), images_b[1].cuda(config['cuda_device']).detach()
            images_a, images_b = images_a[0].cuda(config['cuda_device']).detach(), images_b[0].cuda(config['cuda_device']).detach()

            print("Iteration: " + str(iterations + 1) + "/" + str(max_iter) + " Elapsed time " + str(time.time()-t)[:5])
            t = time.time()

            if iterations > max_iter:
                sys.exit('Finish training')

            # Main training code
            config['iteration'] = iterations
            # numberOf_dis_relative_iteration = config['dis']['numberOf_dis_relative_iteration'] if config['dis']['numberOf_dis_relative_iteration'] > 0 else 1
            # if dis_iter < numberOf_dis_relative_iteration:  # training the discriminetor multiple times for each generator update
            #     dis_iter += 1
            #     print('multi')
            #     trainer.dis_update(images_a, images_b, config)
            #     continue
            # else:
            #     print('single')
            #     trainer.dis_update(images_a, images_b, config)
            #     dis_iter = 1
            
            trainer.dis_update(images_a, images_b, attrs_a, attrs_b, config)

            if config['council']['numberOfCouncil_dis_relative_iteration'] > 0:
                trainer.dis_council_update(images_a, images_b, attrs_a, attrs_b, config)  # the multiple iterating happens inside dis_council_update

            trainer.gen_update(images_a, images_b, config, attrs_a, attrs_b, iterations)
            torch.cuda.synchronize(device=config['cuda_device'])
            iterations += 1

            # write training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                write_loss(iterations, trainer, train_writer)
            # test FID
            if config['misc']['do_test_Fid'] and (iterations + 1) % config['misc']['test_Fid_iter'] == 0:
                if config['do_a2b']:
                    tmp_path_im_a2b = image_directory + '/tmp/a2b'
                    if not os.path.exists(tmp_path_im_a2b):
                        print("Creating directory: {}".format(tmp_path_im_a2b))
                        os.makedirs(tmp_path_im_a2b)
                    filelist2 = [f for f in os.listdir(tmp_path_im_a2b) if f.endswith(".jpg")]
                    for f2 in filelist2:
                        os.remove(os.path.join(tmp_path_im_a2b, f2))


                if config['do_a2b']:
                    tmp_images_a = test_loader_a[0].dataset[0][0].cuda(config['cuda_device']).unsqueeze(0)

                ind_a2b = 0
                ind_b2a = 0
                for k in tqdm(range(1, config['misc']['test_Fid_num_of_im']), desc='Creating images for tests'):
                    c_ind = np.random.randint(config['council']['council_size'])
                    if config['do_a2b']:
                        tmp_images_a = test_loader_a[0].dataset[k][0].cuda(config['cuda_device']).unsqueeze(0)
                        tmp_attrs_a = torch.tensor([test_loader_a[0].dataset[k][1]], dtype=torch.int).cuda(config['cuda_device'])
                        
                        styles = torch.randn(tmp_images_a.shape[0], config['gen']['style_dim'], 1, 1).cuda(config['cuda_device'])
                        
                        tmp_res_imges_a2b = trainer.sample(x_a=tmp_images_a, x_b=None, attrs_a=tmp_attrs_a, s_a=styles, s_b=styles)
                        tmp_res_imges_a2b = tmp_res_imges_a2b[2][c_ind].unsqueeze(0)
                        for tmp_res_imges_a2b_t in tmp_res_imges_a2b:
                            vutils.save_image(tmp_res_imges_a2b_t, tmp_path_im_a2b + '/' + str(ind_a2b) + '.jpg')
                            ind_a2b += 1


                if config['do_a2b']:
                    dataset_for_fid_B = os.path.join(config['data_root'], 'testB')
                    tmp_path_a2b_save_stat = dataset_for_fid_B
                    if os.path.exists(tmp_path_a2b_save_stat + '/m1'):
                        with open(tmp_path_a2b_save_stat + '/m1', 'rb') as f:
                            m1_1_a2b = pickle.load(f)
                    if os.path.exists(tmp_path_a2b_save_stat + '/s1'):
                        with open(tmp_path_a2b_save_stat + '/s1', 'rb') as f:
                            s1_1_a2b = pickle.load(f)

                    if m1_1_a2b is None or s1_1_a2b is None:
                        print('fid test initialization')
                        m1_1_a2b, s1_1_a2b = test_fid(dataset_for_fid_B, tmp_path_im_a2b, iterations, train_writer, 'B', retun_m1_s1=True, batch_size=10)
                        if m1_1_a2b is not None and s1_1_a2b is not None:
                            with open(tmp_path_a2b_save_stat + '/m1', 'wb') as f:
                                pickle.dump(m1_1_a2b, f)
                            with open(tmp_path_a2b_save_stat + '/s1', 'wb') as f:
                                pickle.dump(s1_1_a2b, f)
                    else:
                        _ = test_fid(dataset_for_fid_B, tmp_path_im_a2b, iterations, train_writer, 'B', m1_1_a2b, s1_1_a2b)




            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b, test_display_attrs_a, test_display_attrs_b, iterations)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b, train_display_attrs_a, train_display_attrs_b, iterations)
                test_gen_a2b_im, test_gen_b2a_im = write_2images(test_image_outputs,
                                                                 display_size * config['council']['council_size'],
                                                                 image_directory, 'test_%08d' % (iterations + 1), do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])
                train_gen_a2b_im, train_gen_b2a_im = write_2images(train_image_outputs,
                                                                   display_size * config['council']['council_size'],
                                                                   image_directory, 'train_%08d' % (iterations + 1), do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])

                if config['do_a2b']:
                    train_writer.add_image('a2b/train', train_gen_a2b_im, iterations)
                    train_writer.add_image('a2b/test', test_gen_a2b_im, iterations)
                    

                    
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images', do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b, train_display_attrs_a, train_display_attrs_b, iterations)

                write_2images(image_outputs, display_size * config['council']['council_size'], image_directory,
                              'train_current', do_a2b=config['do_a2b'], do_b2a=config['do_b2a'])

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                try:
                    trainer.save(checkpoint_directory, iterations)
                except Exception as e:
                    print('================================= Faild to save check avileble memory =================================')
                    print(e)
                    input("Clear space and press enter to retry ....")
                    print("retrying to save...")
                    trainer.save(checkpoint_directory, iterations)
            
            trainer.update_learning_rate()

except Exception as e:
    print('Error')
    print('-' * 60)
    traceback.print_exc(file=sys.stdout)
    print('-' * 60)
    print(e)
    print(colored('Training STOPED!', color='red', attrs=['underline', 'bold', 'blink', 'reverse']))
