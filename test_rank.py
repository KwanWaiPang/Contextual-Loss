import os
import sys
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import rgb2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt, is_train = False)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [%s]: %d' % (dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('Testing [%s]...' % test_set_name)
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)


    for data in test_loader:
        # need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

        model.feed_data(data, volatile=True, need_img2=False)
        img_path = data['img1_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        f = open(os.path.join(dataset_dir,'predict_score.txt'),'a')

        model.test()  # test
        visuals = model.get_current_visuals()
        predict_score1 = visuals['predict_score1'].numpy()
        f.write('%s  %f\n'%(img_name+'.png',predict_score1))
        f.close()
        # sr_img = util.tensor2img_np(visuals['SR'])  # uint8


