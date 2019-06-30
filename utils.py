import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d

# session config
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        per_process_gpu_memory_fraction=0.15,
        allow_growth=False
    )
)

# calculate total parameters
def cal_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return print('Total params: %d ' % total_parameters)

# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.bitwise_and(im1, im2).sum()) / np.double(np.bitwise_or(im1, im2).sum())

# calculate L1
def L1norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean(abs(im1 - im2)))

def matplotlib_plt(X, filename):
    fig = plt.figure()
    plt.title('latent distribution')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('dim_1')
    ax.set_ylabel('dim_2')
    ax.set_zlabel('dim_3')
    ax.scatter(X[:,0], X[:,1], X[:,2] , marker="x"
               # , c=y/len(set(y))
    )
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.savefig(filename + "3D/{:03d}.jpg".format(angle))
    # plt.savefig(filename)
    # plt.show()

def visualize_slices(X, Xe, outdir):
    # plot reconstruction
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))
    for i in range(10):
        minX = np.min(X[i, :])
        maxX = np.max(X[i, :])
        axes[0, i].imshow(X[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=minX, vmax=maxX,
                          interpolation='none')
        axes[0, i].set_title('original %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        minXe = np.min(Xe[i, :])
        maxXe = np.max(Xe[i, :])
        axes[1, i].imshow(Xe[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=minXe, vmax=maxXe,
                          interpolation='none')
        axes[1, i].set_title('reconstruction %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "reconstruction.png")
    # plt.show()

def display_center_slices(case, size, num_data, outdir):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    # axial
    fig, axes = plt.subplots(ncols=num_data, nrows=1, figsize=(num_data, 2))
    for i in range(num_data):
        axes[i].imshow(case[i, 3, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('image%d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "/interpolation.png")