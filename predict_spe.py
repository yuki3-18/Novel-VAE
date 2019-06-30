'''
# Generate image and evalute the performance by Specificity
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
import utils
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #for windows

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("train_data_txt", "E:/git/beta-VAE/input/CT/shift/train.txt", "train data txt")
    flags.DEFINE_string("ground_truth_txt", "E:/git/beta-VAE/input/CT/shift/test.txt","i1")
    flags.DEFINE_string("model1", 'D:/vae_result/n1/z6/beta_1/model/model_{}'.format(997500), "i2")
    flags.DEFINE_string("model2", 'D:/vae_result/n1+n2/all/sig/beta_1/model/model_{}'.format(197500), "i3")
    flags.DEFINE_string("outdir", "D:/vae_result/n1+n2/all/sig/beta_1/spe/", "i4")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_generate", 5000, "number of generate data")
    flags.DEFINE_integer("num_of_test", 600, "number of test data")
    flags.DEFINE_integer("num_of_train", 1804, "number of train data")
    flags.DEFINE_integer("batch_size", 1, "batch size")
    flags.DEFINE_integer("latent_dim", 6, "latent dim")
    flags.DEFINE_list("image_size", [9 * 9 * 9], "image size")
    flags.DEFINE_boolean("const_bool", False, "if there is sigmoid in front of last output")
    FLAGS = flags.FLAGS

    # check folder
    if not (os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir + 'spe1/')
        os.makedirs(FLAGS.outdir + 'spe2/')
        os.makedirs(FLAGS.outdir + 'spe_all/')

    # read list
    test_data_list = io.load_list(FLAGS.ground_truth_txt)
    train_data_list = io.load_list(FLAGS.train_data_txt)

    # test step
    test_step = FLAGS.num_of_generate // FLAGS.batch_size
    if FLAGS.num_of_generate % FLAGS.batch_size != 0:
        test_step += 1

    # load train data
    train_set = tf.data.TFRecordDataset(train_data_list)
    train_set = train_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                            num_parallel_calls=os.cpu_count())
    train_set = train_set.batch(FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

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

        # set network
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
            'is_res': False,
            'is_constraints': FLAGS.const_bool
        }
        VAE_2 = Variational_Autoencoder(**kwargs_2)

        sess.run(init_op)

        # testing
        VAE.restore_model(FLAGS.model1)
        VAE_2.restore_model(FLAGS.model2)

        tbar = tqdm(range(FLAGS.num_of_generate), ascii=True)
        specificity = []
        spe_mean = []
        generate_data = []
        generate_data2 = []
        ori = []
        latent_space = []
        latent_space2 = []

        patch_side = 9

        for i in range(FLAGS.num_of_train):
            train_data_batch = sess.run(train_data)
            z = VAE.plot_latent(train_data_batch)
            z2=VAE_2.plot_latent(train_data_batch)
            z = z.flatten()
            z2=z2.flatten()
            latent_space.append(z)
            latent_space2.append(z2)

        mu = np.mean(latent_space, axis=0)
        var = np.var(latent_space, axis=0)
        mu2 = np.mean(latent_space2, axis=0)
        var2 = np.var(latent_space2, axis=0)

        for i in range(FLAGS.num_of_test):
            test_data_batch = sess.run(test_data)
            ori_single = test_data_batch
            ori_single = ori_single[0, :]
            ori.append(ori_single)

        file_spe1 = open(FLAGS.outdir + 'spe1/list.txt', 'w')
        file_spe2 = open(FLAGS.outdir + 'spe2/list.txt', 'w')
        file_spe_all = open(FLAGS.outdir + 'spe_all/list.txt', 'w')

        for j in tbar:
            sample_z = np.random.normal(mu, var, (1, FLAGS.latent_dim))
            sample_z2 = np.random.normal(mu2, var2, (1, 8))
            generate_data_single = VAE.generate_sample(sample_z)
            if FLAGS.const_bool is False:
                generate_data_single2 = VAE_2.generate_sample(sample_z2)
                generate_data_single = generate_data_single[0, :]
                generate_data_single2 = generate_data_single2[0, :]
                generate_data.append(generate_data_single)
                generate_data2.append(generate_data_single2)
                gen = np.reshape(generate_data_single, [patch_side, patch_side, patch_side])
                gen2 = np.reshape(generate_data_single2, [patch_side, patch_side, patch_side])
                generate_data_single_all = generate_data_single + generate_data_single2
                gen_all = gen + gen2

            if FLAGS.const_bool is True:
                generate_data_single_all = VAE_2.generate_sample2(sample_z2, generate_data_single)
                generate_data_single = generate_data_single[0, :]
                generate_data_single_all = generate_data_single_all[0, :]
                generate_data.append(generate_data_single)
                generate_data2.append(generate_data_single_all)
                gen = np.reshape(generate_data_single, [patch_side, patch_side, patch_side])
                gen_all = np.reshape(generate_data_single_all, [patch_side, patch_side, patch_side])
                generate_data_single2 = generate_data_single_all - generate_data_single
                gen2 = gen_all - gen

            # EUDT
            gen_image = sitk.GetImageFromArray(gen)
            gen_image.SetSpacing([0.885, 0.885, 1])
            gen_image.SetOrigin([0, 0, 0])

            gen2_image = sitk.GetImageFromArray(gen2)
            gen2_image.SetSpacing([0.885, 0.885, 1])
            gen2_image.SetOrigin([0, 0, 0])

            gen_all_image = sitk.GetImageFromArray(gen_all)
            gen_all_image.SetSpacing([0.885, 0.885, 1])
            gen_all_image.SetOrigin([0, 0, 0])

            # calculation
            case_min_specificity = 1.0
            for image_index in range(FLAGS.num_of_test):
                specificity_tmp = utils.L1norm(ori[image_index] ,generate_data_single_all)
                if specificity_tmp < case_min_specificity:
                    case_min_specificity = specificity_tmp

            specificity.append([case_min_specificity])
            spe = np.mean(specificity)
            spe_mean.append(spe)


            io.write_mhd_and_raw(gen_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'spe1','spe1_{}'.format(j + 1))))
            io.write_mhd_and_raw(gen2_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'spe2','spe2_{}'.format(j + 1))))
            io.write_mhd_and_raw(gen_all_image, '{}.mhd'.format(os.path.join(FLAGS.outdir, 'spe_all', 'spe_all_{}'.format(j + 1))))
            file_spe1.write('{}.mhd'.format(os.path.join(FLAGS.outdir, 'spe1', 'spe1_{}'.format(j + 1))) + "\n")
            file_spe2.write('{}.mhd'.format(os.path.join(FLAGS.outdir, 'spe2', 'spe2_{}'.format(j + 1))) + "\n")
            file_spe_all.write('{}.mhd'.format(os.path.join(FLAGS.outdir, 'spe_all', 'spe_all_{}'.format(j + 1))) + "\n")

    file_spe1.close()
    file_spe2.close()
    file_spe_all.close()

    print('specificity = %f' % np.mean(specificity))
    np.savetxt(os.path.join(FLAGS.outdir, 'specificity.csv'), specificity, delimiter=",")

    # spe graph
    plt.plot(spe_mean)
    plt.grid()
    # plt.show()
    plt.savefig(FLAGS.outdir + "Specificity.png")


# # load tfrecord function
def _parse_function(record, image_size=[9, 9, 9]):
    keys_to_features = {
        'img_raw': tf.FixedLenFeature(np.prod(image_size), tf.float32),
    }
    parsed_features = tf.parse_single_example(record, keys_to_features)
    image = parsed_features['img_raw']
    image = tf.reshape(image, image_size)
    return image

if __name__ == '__main__':
    main()