# -*- coding: utf-8 -*-
import os

## GAN Variants
from AE_GAN import AE_GAN
from CAE_CGAN import CAE_CGAN

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='AE_GAN',
                        choices=['AE_GAN', 'CAE_CGAN'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=62, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --z_dim
    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')

    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        if args.gan_type == 'AE_GAN':
            gan = AE_GAN(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim,dataset_name=args.dataset,
                         checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        elif args.gan_type == 'CAE_CGAN':
            gan = CAE_CGAN(sess, epoch=args.epoch, batch_size=args.batch_size, z_dim=args.z_dim,dataset_name=args.dataset,
                           checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)
        else:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()

        gan.train_check()

        # visualize learned generator
        gan.visualize_results(args.epoch - 1)
        print(" [*] Testing finished!")


if __name__ == '__main__':
    main()
