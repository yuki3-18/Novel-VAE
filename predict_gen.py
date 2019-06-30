'''
# Reconstruct image and evalute the performance by Generalization
# Author: Yuki Saeki
# Reference: "https://github.com/Silver-L/beta-VAE"
'''

import tensorflow as tf
import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import csv
import dataIO as io
from network import *
from model import Variational_Autoencoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("test_data_txt", "E:/git/beta-VAE/input/CT/shift/test.txt", "i1")
    flags.DEFINE_string("model1", 'D:/vae_result/n1+n2/all/fine/beta_1/model/model_{}'.format(198500), "i2")
    flags.DEFINE_string("model2", 'D:/vae_result/n1+n2/all/fine/beta_1/model2/model_{}'.format(198500), "i3")
    flags.DEFINE_string("outdir", "D:/vae_result/n1+n2/all/fine/beta_1/gen/", "i4")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_test", 600, "number of test data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 6, "latent dim")
    flags.DEFINE_list("image_size", [9*9*9], "image size")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir + 'ori/')
        os.makedirs(FLAGS.outdir + 'preds/')
        os.makedirs(FLAGS.outdir + 'rec/')


    # read list
    test_data_list = io.load_list(FLAGS.test_data_txt)


    # test step
    test_step = FLAGS.num_of_test // FLAGS.batch_size
    if FLAGS.num_of_test % FLAGS.batch_size != 0:
        test_step += 1

    # load test data
    test_set = tf.data.TFRecordDataset(test_data_list)
    test_set = test_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                            num_parallel_calls=os.cpu_count())
    test_set = test_set.batch(FLAGS.batch_size)
    test_iter = test_set.make_one_shot_iterator()
    test_data = test_iter.get_next()

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config) as sess:

        sess.run(init_op)

        # set network
        kwargs = {
            'sess': sess,
            'outdir': FLAGS.outdir,
            'beta': FLAGS.beta,
            'latent_dim': FLAGS.latent_dim,
            'batch_size': FLAGS.batch_size,
            'image_size': FLAGS.image_size,
            'encoder': encoder_mlp,
            'decoder': decoder_mlp,
            'is_res': False
        }
        VAE = Variational_Autoencoder(**kwargs)
        kwargs_2 = {
            'sess': sess,
            'outdir': FLAGS.outdir,
            'beta': FLAGS.beta,
            'latent_dim': 8,
            'batch_size': FLAGS.batch_size,
            'image_size': FLAGS.image_size,
            'encoder': encoder_mlp2,
            'decoder': decoder_mlp_tanh,
            'is_res': True,
            'is_constraints': False
        }
        VAE_2 = Variational_Autoencoder(**kwargs_2)


        # testing
        VAE.restore_model(FLAGS.model1)
        VAE_2.restore_model(FLAGS.model2)

        tbar = tqdm(range(test_step), ascii=True)
        preds = []
        ori = []
        rec = []

        for k in tbar:
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            preds_single = VAE.reconstruction_image(ori_single)
            rec_single = VAE_2.reconstruction_image2(ori_single, preds_single)
            preds_single = preds_single[0, :]
            ori_single = ori_single[0, :]
            rec_single = rec_single[0, :]
            preds.append(preds_single)
            ori.append(ori_single)
            rec.append(rec_single)

        patch_side = 9

        preds = np.reshape(preds, [FLAGS.num_of_test, patch_side, patch_side, patch_side])
        ori = np.reshape(ori, [FLAGS.num_of_test, patch_side, patch_side, patch_side])
        rec = np.reshape(rec, [FLAGS.num_of_test, patch_side, patch_side, patch_side])

        # label
        generalization_single = []
        file_ori = open(FLAGS.outdir + 'ori/list.txt', 'w')
        file_preds = open(FLAGS.outdir + 'preds/list.txt', 'w')
        file_rec = open(FLAGS.outdir + 'rec/list.txt', 'w')

        for j in range(len(preds)):

            # EUDT
            ori_image = sitk.GetImageFromArray(ori[j])
            ori_image.SetOrigin([0, 0, 0])
            ori_image.SetSpacing([0.885,0.885,1])

            preds_image = sitk.GetImageFromArray(preds[j])
            preds_image.SetOrigin([0, 0, 0])
            preds_image.SetSpacing([0.885,0.885,1])

            rec_image = sitk.GetImageFromArray(rec[j])
            rec_image.SetOrigin([0, 0, 0])
            rec_image.SetSpacing([0.885,0.885,1])

            # output image
            io.write_mhd_and_raw(ori_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'ori','ori_{}'.format(j + 1))))
            io.write_mhd_and_raw(preds_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'preds','preds_{}'.format(j + 1))))
            io.write_mhd_and_raw(rec_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'rec', 'rec_{}'.format(j + 1))))
            file_ori.write('{}.mhd'.format(os.path.join(FLAGS.outdir, 'ori', 'ori_{}'.format(j + 1))) + "/n")
            file_rec.write('{}.mhd'.format(os.path.join(FLAGS.outdir, 'preds', 'preds_{}'.format(j + 1))) + "/n")
            file_rec.write('{}.mhd'.format(os.path.join(FLAGS.outdir, 'rec', 'rec_{}'.format(j + 1))) + "/n")

            generalization_single.append(utils.L1norm(ori[j], rec[j]))

        file_ori.close()
        file_preds.close()
        file_rec.close()

    generalization = np.average(generalization_single)
    print('generalization = %f' % generalization)

    np.savetxt(os.path.join(FLAGS.outdir, 'generalization.csv'), generalization_single, delimiter=",")

    # plot reconstruction
    a_X = ori[:, 4, :]
    a_Xe = rec[:, 4, :]
    c_X = ori[:, :, 4, :]
    c_Xe = rec[:, :, 4, :]
    s_X = ori[:, :, :, 4]
    s_Xe = rec[:, :, :, 4]
    utils.visualize_slices(a_X, a_Xe, FLAGS.outdir + "axial_")
    utils.visualize_slices(c_X, c_Xe, FLAGS.outdir + "coronal_")
    utils.visualize_slices(s_X, s_Xe, FLAGS.outdir + "sagital_")

# # load tfrecord function
def _parse_function(record, image_size=[9 * 9 * 9]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image


if __name__ == '__main__':
    main()