'''
# Beta Variational AutoEncoder Trainer
# Author: Yuki Saeki
# Reference: "https://github.com/Silver-L/beta-VAE"
'''
import tensorflow as tf
import os, random
import argparse
import numpy as np
from tqdm import tqdm
import dataIO as io
from network import *
from model import Variational_Autoencoder
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          #for windows
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("train_data_txt", "./train.txt", "train data txt")
    flags.DEFINE_string("val_data_txt", "./val.txt", "validation data txt")
    flags.DEFINE_string("outdir", "./output/", "outdir")
    flags.DEFINE_float("beta", 1, "hyperparameter beta")
    flags.DEFINE_integer("num_of_val", 600, "number of validation data")
    flags.DEFINE_integer("batch_size", 30, "batch size")
    flags.DEFINE_integer("num_iteration", 500001, "number of iteration")
    flags.DEFINE_integer("save_loss_step", 200, "step of save loss")
    flags.DEFINE_integer("save_model_step", 500, "step of save model and validation")
    flags.DEFINE_integer("shuffle_buffer_size", 1000, "buffer size of shuffle")
    flags.DEFINE_integer("latent_dim", 6, "latent dim")
    flags.DEFINE_list("image_size", [9*9*9], "image size")
    flags.DEFINE_string("model", './model/model_{}', "pre training model1")
    flags.DEFINE_string("model2", './model/model_{}', "pre training model2")
    flags.DEFINE_boolean("is_n1_opt", True, "n1_opt")
    FLAGS = flags.FLAGS
    
    # check folder
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard', 'train'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard', 'train'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard', 'val'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard', 'val'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard', 'rec'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard', 'rec'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'tensorboard', 'kl'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'tensorboard', 'kl'))
    if not (os.path.exists(os.path.join(FLAGS.outdir, 'model'))):
        os.makedirs(os.path.join(FLAGS.outdir, 'model'))

    # read list
    train_data_list = io.load_list(FLAGS.train_data_txt)
    val_data_list = io.load_list(FLAGS.val_data_txt)

    # shuffle list
    random.shuffle(train_data_list)
    # val step
    val_step = FLAGS.num_of_val // FLAGS.batch_size
    if FLAGS.num_of_val % FLAGS.batch_size != 0:
        val_step += 1

    # load train data and validation data
    train_set = tf.data.Dataset.list_files(train_data_list)
    train_set = train_set.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=6))
    # train_set = tf.data.TFRecordDataset(train_data_list)
    train_set = train_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                              num_parallel_calls=os.cpu_count())
    train_set = train_set.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    train_set = train_set.repeat()
    train_set = train_set.batch(FLAGS.batch_size)
    train_iter = train_set.make_one_shot_iterator()
    train_data = train_iter.get_next()

    val_set = tf.data.Dataset.list_files(val_data_list)
    val_set = val_set.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=os.cpu_count()))
    # val_set = tf.data.TFRecordDataset(val_data_list)
    val_set = val_set.map(lambda x: _parse_function(x, image_size=FLAGS.image_size),
                          num_parallel_calls=os.cpu_count())
    val_set = val_set.repeat()
    val_set = val_set.batch(FLAGS.batch_size)
    val_iter = val_set.make_one_shot_iterator()
    val_data = val_iter.get_next()

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config) as sess:
    # with tf.Session() as sess:
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
            'is_constraints': False,
            # 'keep_prob': 0.5
        }

        VAE_2 = Variational_Autoencoder(**kwargs_2)
        # print parmeters
        utils.cal_parameter()

        # prepare tensorboard
        writer_train = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'train'), sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'val'))
        writer_rec = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'rec'))
        writer_kl = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'tensorboard', 'kl'))

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        # use pre trained model
        # ckpt_state = tf.train.get_checkpoint_state(FLAGS.model)
        #
        # if ckpt_state:
        #     restore_model = ckpt_state.model_checkpoint_path
        #     # VAE.restore_model(FLAGS.model+'model_{}'.format(FLAGS.itr))
        VAE.restore_model(FLAGS.model)
        if FLAGS.is_n1_opt == True:
            VAE_2.restore_model(FLAGS.model2)

        # training
        tbar = tqdm(range(FLAGS.num_iteration), ascii=True)
        for i in tbar:
            train_data_batch = sess.run(train_data)
            if FLAGS.is_n1_opt == True:
                VAE.update(train_data_batch)

            output1 = VAE.reconstruction_image(train_data_batch)

            train_loss, rec_loss, kl_loss = VAE_2.update2(train_data_batch, output1)

            if i % FLAGS.save_loss_step is 0:
                s = "Loss: {:.4f}, rec_loss: {:.4f}, kl_loss: {:.4f}".format(train_loss, rec_loss, kl_loss)
                tbar.set_description(s)
                summary_train_loss = sess.run(merge_op, {value_loss: train_loss})
                writer_train.add_summary(summary_train_loss, i)

                summary_rec_loss = sess.run(merge_op, {value_loss: rec_loss})
                summary_kl_loss = sess.run(merge_op, {value_loss: kl_loss})
                writer_rec.add_summary(summary_rec_loss, i)
                writer_kl.add_summary(summary_kl_loss, i)


            if i % FLAGS.save_model_step is 0:
                # save model
                VAE.save_model(i)
                VAE_2.save_model2(i)

                # validation
                val_loss = 0.
                for j in range(val_step):
                    val_data_batch = sess.run(val_data)
                    val_data_batch_output1 = VAE.reconstruction_image(val_data_batch)

                    val_loss += VAE_2.validation2(val_data_batch, val_data_batch_output1)
                val_loss /= val_step

                summary_val = sess.run(merge_op, {value_loss: val_loss})
                writer_val.add_summary(summary_val, i)

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
