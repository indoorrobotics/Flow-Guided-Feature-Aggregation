# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import sys
print sys.path
import _init_paths

import cv2
import argparse
import os
import sys
import time
import logging
from config.config import config, update_config
from os.path import join

def parse_args():
    parser = argparse.ArgumentParser(description='Test a R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger

def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    output_dir = "/tmp/res"
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    sets = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test_sets/ImageSets"
    for file_name in os.listdir(sets):
        if "_val" in file_name:
            continue

        maps = []
        logger.info("About to test with images:" + file_name)
        print ("About to test with images:" + file_name)
        for epoc in range(1, 30):
            res = test_rcnn(config, config.dataset.dataset, file_name.replace(".txt", ""), config.dataset.root_path, config.dataset.dataset_path, config.dataset.motion_iou_path,
                      ctx, join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), epoc,
                      args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path,
                      enable_detailed_eval=config.dataset.enable_detailed_eval)
        with open(join(output_dir, file_name), "a") as f:
            f.write('epoc: %s res: %s' % (res, epoc))
        maps.append(res)
        print maps

if __name__ == '__main__':
    main()
