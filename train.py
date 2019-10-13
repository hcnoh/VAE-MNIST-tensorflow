import os
import pickle

import numpy as np
import tensorflow as tf

from data.mnist import Mnist
from models.vae import VAE


LATENT_DEPTH = 20

BATCH_SIZE = 32
NUM_EPOCHS = 100

MODEL_SAVEL_DIR = "../../learned-models/variational-autoencoder-mnist"
MODEL_NAME = "variational-autoencoder"


def main():
    model_ckpt_name = "%s-model.ckpt" % MODEL_NAME
    model_spec_name = "%s-model-spec.json" % MODEL_NAME
    model_rslt_name = "%s-results.pickle" % MODEL_NAME

    model_save_path = os.path.join(MODEL_SAVEL_DIR, MODEL_NAME)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_ckpt_path = os.path.join(model_save_path, model_ckpt_name)
    model_spec_path = os.path.join(model_save_path, model_spec_name)
    model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    loader = Mnist()

    features = np.vstack([loader.train_features, loader.test_features])

    num_sets = loader.num_train_sets + loader.num_test_sets
    feature_depth = loader.feature_depth
    feature_shape = loader.feature_shape
    
    x = tf.placeholder(dtype=tf.float32, shape=[None, feature_depth])
    eps = tf.placeholder(dtype=tf.float32, shape=[None, LATENT_DEPTH])

    vae = VAE(x, eps, LATENT_DEPTH, feature_depth)

    encoder_loss = tf.reduce_mean(vae.get_encoder_loss())
    decoder_loss = tf.reduce_mean(vae.get_decoder_loss())

    loss = encoder_loss + decoder_loss
    opt = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=1000)

    steps_per_epoch = num_sets // BATCH_SIZE
    train_steps = steps_per_epoch * NUM_EPOCHS

    encoder_losses = []
    decoder_losses = []
    losses = []
    encoder_losses_epoch = []
    decoder_losses_epoch = []
    losses_epoch = []
    fs = []
    for i in range(train_steps):
        epoch = i // steps_per_epoch

        idxes = np.random.choice(num_sets, BATCH_SIZE, replace=False)
        features_i = features[idxes]
        eps_i = np.random.normal(size=[BATCH_SIZE, LATENT_DEPTH])

        # if i % 1 == 0:
        #     loss_g_i, _ = sess.run(
        #         [loss_g, opt_g], feed_dict={x: features_i, z: z_i}
        #     )
        # if i % 2 == 0:
        #     loss_d_i, _ = sess.run(
        #         [loss_d, opt_d], feed_dict={x: features_i, z: z_i}
        #     )
        encoder_loss_i, decoder_loss_i, loss_i, _ = sess.run(
            [encoder_loss, decoder_loss, loss, opt], feed_dict={x: features_i, eps: eps_i}
        )
        
        encoder_losses.append(encoder_loss_i)
        decoder_losses.append(decoder_loss_i)
        losses.append(loss_i)

        # if i % steps_per_epoch == 0:
        if i % 100 == 0:
            temp_idx = 0
            f_eps = sess.run(
                vae.f_eps[temp_idx], feed_dict={eps: eps_i}
            )

            encoder_loss_epoch = np.mean(encoder_losses[-steps_per_epoch:])
            decoder_loss_epoch = np.mean(decoder_losses[-steps_per_epoch:])
            loss_epoch = np.mean(losses[-steps_per_epoch:])

            print(
                "Epoch: %i,  Encoder Loss: %f,  Decoder Loss: %f" % (
                    epoch, encoder_loss_epoch, decoder_loss_epoch
                )
            )

            encoder_losses_epoch.append(encoder_loss_epoch)
            decoder_losses_epoch.append(decoder_loss_epoch)
            losses_epoch.append(loss_epoch)

            fs.append(f_eps)

            # saver.save(sess, model_ckpt_path, global_step=i)

            with open(model_rslt_path, "wb") as f:
                pickle.dump(
                    (
                        encoder_losses_epoch,
                        decoder_losses_epoch,
                        losses_epoch,
                        fs
                    ),
                    f
                )


if __name__ == "__main__":
    main()