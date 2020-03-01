
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mobilenet_v3_block import BottleNeck, h_swish
from yolov3_layer_utils import upsample_layer, yolo_conv2d, yolo_block
import numpy as np
import sys
import os
sys.path.append("../")
from utils.utils import resize_image
from utils.visualize import display_instances

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Lambda

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class EpochRecord(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(EpochRecord, self).__init__()
        self.name = name

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.name+"/epoch.txt"):
            file = open(self.name+"/epoch.txt", 'w')
            file.write("0")
            file.close()
        file = open(self.name+"/epoch.txt", 'r')
        epoch = int(str(file.readline()))
        file.close()
        epoch += 1
        epoch = str(epoch)
        file = open(self.name + "/epoch.txt", 'w')
        file.write(epoch)
        file.close()


class CentLoss(tf.keras.layers.Layer):
    def __init__(self, batch_size, num_class, decay, stride, **kwargs):
        super(CentLoss, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_class = num_class
        self.decay = decay
        self.stride = stride

    def call(self, inputs, **kwargs):
        center, preg, fpn, ground_truth = inputs
        losses = self._centernet_loss(center, preg, fpn, ground_truth)

        self.add_loss(losses[4], inputs=True)
        self.add_metric(losses[0], aggregation="mean", name="center")
        self.add_metric(losses[1], aggregation="mean", name="iou")
        self.add_metric(losses[2], aggregation="mean", name="size")
        self.add_metric(losses[3], aggregation="mean", name="seg")
        return losses[4]

    def _centernet_loss(self, keypoints, preg, fpn, ground_truth):

        total_loss = []
        for i in range(self.batch_size):
            loss = self._compute_one_image_loss(keypoints[i, ...], preg[i, ...], fpn[i, ...], ground_truth[i, ...])
            total_loss.append(loss)
        mean_loss = tf.reduce_sum(total_loss, axis=0) / self.batch_size
        return mean_loss

    def _compute_one_image_loss(self, gravity_pred, dist_pred, heatmap_pred, ground_truth):

        h = tf.shape(gravity_pred)[0]
        w = tf.shape(gravity_pred)[1]
        dist_pred_t = dist_pred[..., 0]  # (y, x)
        dist_pred_l = dist_pred[..., 1]
        dist_pred_b = dist_pred[..., 2]
        dist_pred_r = dist_pred[..., 3]

        dist_t = ground_truth[..., 0]
        dist_l = ground_truth[..., 1]
        dist_b = ground_truth[..., 2]
        dist_r = ground_truth[..., 3]


        inter_width = tf.minimum(dist_l, dist_pred_l) + tf.minimum(dist_r, dist_pred_r)
        inter_height = tf.minimum(dist_t, dist_pred_t) + tf.minimum(dist_b, dist_pred_b)
        inter_area = inter_width * inter_height
        union_area = (dist_l + dist_r) * (dist_t + dist_b) + (dist_pred_l + dist_pred_r) * (
                dist_pred_t + dist_pred_b) - inter_area
        iou = inter_area / (union_area + 1e-12)

        # iou_loss
        iou_reduction = ground_truth[..., 4]
        ioc = tf.cast(iou_reduction > 0.0, tf.float32)
        iou_loss = tf.reduce_sum(-tf.math.log(iou + 1e-12) * ioc * (iou_reduction + 1.0))

        # size_loss
        size_center = tf.cast(iou_reduction >= 1.0, tf.float32)
        size_loss = 10 * tf.reduce_sum(-tf.math.log(iou + 1e-12) * size_center)

        # center_loss
        gt_center = ground_truth[..., 5:5 + self.num_class]
        gt_num = tf.reduce_sum(gt_center)
        reduction = ground_truth[..., 5 + self.num_class:5 + 2 * self.num_class]
        center_pos_loss = - tf.pow(1. - tf.sigmoid(gravity_pred), 2.) * tf.math.log_sigmoid(gravity_pred) * gt_center
        center_neg_loss = -tf.pow(1. - reduction, 4) * tf.pow(tf.sigmoid(gravity_pred), 2.) * (
                    -gravity_pred + tf.math.log_sigmoid(gravity_pred)) * (1. - gt_center)
        center_loss = tf.reduce_sum(center_pos_loss) + tf.reduce_sum(center_neg_loss)

        # seg_loss
        gt_seg = ground_truth[..., 5 + 2 * self.num_class:5 + 3 * self.num_class]
        seg_pos_loss = - 100 * tf.pow(1. - tf.sigmoid(heatmap_pred), 2.) * tf.math.log_sigmoid(heatmap_pred) * gt_seg
        seg_neg_loss = - 100 * tf.pow(tf.sigmoid(heatmap_pred), 2.) * (
                    -heatmap_pred + tf.math.log_sigmoid(heatmap_pred)) * (1. - gt_seg)

        if tf.reduce_sum(gt_seg) != 0:
            seg_loss = tf.reduce_sum(seg_pos_loss) / tf.reduce_sum(gt_seg) + tf.reduce_sum(seg_neg_loss) / (
                        tf.cast((w * h * self.num_class), tf.float32) - tf.reduce_sum(gt_seg))
            iou_loss = iou_loss / tf.reduce_sum(gt_seg)
            size_loss = size_loss / tf.cast(gt_num, tf.float32)
            center_loss = center_loss / tf.cast(gt_num, tf.float32)
            total_loss = iou_loss + center_loss + seg_loss + size_loss
            return center_loss, iou_loss, seg_loss, size_loss, total_loss
        else:
            return .0, .0, .0, .0, .0


class CenterNet:
    def __init__(self, config, name):
        self.name = name
        self.config = config
        assert config.MODEL in ['train', 'infer']
        self.mode = config.MODEL
        self.data_shape = config.IMAGE_SHAPE
        self.image_size = config.IMAGE_MAX_DIM
        self.stride = config.STRIDE
        self.num_classes = config.NUM_CLASSES
        self.loss_decay = config.LOSS_DECAY
        self.l2_decay = config.L2_DECAY
        self.data_format = config.DATA_FORMAT
        self.batch_size = config.BATCH_SIZE if config.MODEL == 'train' else 1
        self.max_gt_instances = config.MAX_GT_INSTANCES
        self.gt_channel = config.GT_CHANNEL
        self.seg_threshold = config.SEG_THRESHOLD
        #
        self.top_k_results_output = config.DETECTION_MAX_INSTANCES
        self.nms_threshold = config.DETECTION_NMS_THRESHOLD
        self.train_bn = config.TRAIN_BN
        self.box_threshold = config.BOX_THRESHOLD
        #
        self.score_threshold = config.SCORE_THRESHOLD
        self.is_training = True if config.MODEL == 'train' else False

        if not os.path.exists(name):
            os.mkdir(name)
        self.checkpoint_path = name

        if not os.path.exists(name + "/log"):
            os.mkdir(name + "/log")
        self.log_dir = name + "/log"

        if not os.path.exists(name+"/epoch.txt"):
            file = open(name+"/epoch.txt", 'w')
            file.write("0")
            file.close()

        file = open(name + "/epoch.txt", 'r')
        self.pro_epoch = int(str(file.readline()))
        file.close()
        self._define_inputs()
        self._build_graph()
        if self.pro_epoch != 0:
            self.load_weight(self.pro_epoch)

    def _define_inputs(self):
        # model inputs: [images, ground_truth, mask_ground_truth]
        shape = self.data_shape
        self.images = tf.keras.Input(shape=shape, dtype=tf.float32)

        if self.mode == 'train':
            gt_shape = [self.image_size/int(self.stride), self.image_size/int(self.stride), self.gt_channel]
            self.ground_truth = tf.keras.Input(shape=gt_shape, dtype=tf.float32)

    def _build_backbone(self, x):
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding="same")(x)  # 256, 2
        x = BatchNormalization(name='first_bn', epsilon=1e-5)(x)
        x = h_swish(x)
        x = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1, is_se_existing=False, NL="RE", k=3)(x)  # 256
        x = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2, is_se_existing=False, NL="RE", k=3)(x)  # 128, 4

        s_4 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1, is_se_existing=False, NL="RE", k=3)(x)  # 128
        x = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2, is_se_existing=True, NL="RE", k=5)(s_4)  # 64, 8

        x = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)(x)
        s_8 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1, is_se_existing=True, NL="RE", k=5)(x)
        x = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2, is_se_existing=False, NL="HS", k=3)(s_8)  # 32

        x = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)(x)
        x = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)(x)
        x = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1, is_se_existing=False, NL="HS", k=3)(x)
        x = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)(x)
        s_16 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1, is_se_existing=True, NL="HS", k=3)(x)
        x = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2, is_se_existing=True, NL="HS", k=5)(s_16)  # 16

        x = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)(x)
        s_32 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1, is_se_existing=True, NL="HS", k=5)(x)
        # x = Conv2D(filters=960, kernel_size=(1, 1), strides=1, padding="same")(x)
        # x = BatchNormalization(epsilon=1e-5)(x)
        # s_32 = Activation('relu')(x)

        return s_4, s_8, s_16, s_32

    def _fusion_feature(self, s_4, s_8, s_16, s_32):

        s_32 = yolo_conv2d(s_32, 256, 3, 1)  # 16 /32
        s_32 = self._dconv_bn_activation(s_32, 256, 4, 2)  # 32 /16
        concat1 = tf.concat([s_32, s_16], axis=3)
        s_16 = Conv2D(256, (1, 1), padding='same', use_bias=False)(concat1)
        s_16 = BatchNormalization(epsilon=1e-5)(s_16)
        s_16 = Activation('relu')(s_16)
        s_16 = self._SepConv_BN(s_16, 256, 's16_dp1', point_activation=True, epsilon=1e-5)
        s_16 = self._SepConv_BN(s_16, 256, 's16_dp2', point_activation=True, epsilon=1e-5)
        s_16 = self._SepConv_BN(s_16, 256, 's16_dp3', point_activation=True, epsilon=1e-5)
        s_16 = self._SepConv_BN(s_16, 256, 's16_dp4', point_activation=True, epsilon=1e-5)
        s_16 = self._SepConv_BN(s_16, 256, 's16_dp5', point_activation=True, epsilon=1e-5)

        s_16 = self._dconv_bn_activation(s_16, 256, 4, 2)  # 64 /8
        concat2 = tf.concat([s_16, s_8], axis=3)
        s_8 = Conv2D(256, (1, 1), padding='same', use_bias=False)(concat2)
        s_8 = BatchNormalization(epsilon=1e-5)(s_8)
        s_8 = Activation('relu')(s_8)
        s_8 = yolo_conv2d(s_8, 256, 3, 1)
        s_8 = self._SepConv_BN(s_8, 256, 's8_dp1', point_activation=True, epsilon=1e-5)
        s_8 = self._SepConv_BN(s_8, 256, 's8_dp2', point_activation=True, epsilon=1e-5)
        s_8 = self._SepConv_BN(s_8, 256, 's8_dp3', point_activation=True, epsilon=1e-5)
        s_8 = self._SepConv_BN(s_8, 256, 's8_dp4', point_activation=True, epsilon=1e-5)
        s_8 = self._SepConv_BN(s_8, 256, 's8_dp5', point_activation=True, epsilon=1e-5)

        # s_8 = upsample_layer(s_8, out_shape=128)  # 128 /4
        # concat3 = tf.concat([s_8, s_4], axis=3)
        # s_4 = Conv2D(128, (1, 1), padding='same', use_bias=False)(concat3)
        # s_4 = BatchNormalization(epsilon=1e-5)(s_4)
        # s_4 = Activation('relu')(s_4)
        # s_4 = self._SepConv_BN(s_4, 128, 's4_dp1', point_activation=True, epsilon=1e-5)
        # s_4 = self._SepConv_BN(s_4, 128, 's4_dp2', point_activation=True, epsilon=1e-5)
        # s_4 = self._SepConv_BN(s_4, 128, 's4_dp3', point_activation=True, epsilon=1e-5)

        return s_8

    def _detect_head(self, s_8):

        reg = self._conv_bn_activation(s_8, 256, 3, 1)
        reg = self._conv_bn_activation(reg, 256, 3, 1)
        # conv3 = self._conv_bn_activation(conv2, 256, 3, 1)
        # conv4 = self._conv_bn_activation(conv3, 256, 3, 1)
        reg = self._SepConv_BN(reg, 256, 'reg_depth_point_1', point_activation=True, epsilon=1e-5)
        reg = self._SepConv_BN(reg, 256, 'reg_depth_point_2', point_activation=True, epsilon=1e-5)
        reg = self._SepConv_BN(reg, 256, 'reg_depth_point_3', point_activation=True, epsilon=1e-5)
        reg = self._SepConv_BN(reg, 4, 'reg_depth_point_5', epsilon=1e-5)
        reg = tf.exp(reg)
        # size = self._conv_activation(conv4, 4, 3, 1, activation=tf.exp)

        # center and seg
        center = self._SepConv_BN(s_8, 128, 'cent_depth_point_1', point_activation=True, epsilon=1e-5)
        center = self._SepConv_BN(center, 128, 'cent_depth_point_2', point_activation=True, epsilon=1e-5)
        center = self._SepConv_BN(center, 128, 'cent_depth_point_3', point_activation=True, epsilon=1e-5)
        center = self._SepConv_BN(center, 128, 'cent_depth_point_4', point_activation=True, epsilon=1e-5)
        center = self._SepConv_BN(center, 128, 'cent_depth_point_5', point_activation=True, epsilon=1e-5)
        center = self._SepConv_BN(center, self.num_classes, 'cent', epsilon=1e-5)

        seg = self._SepConv_BN(s_8, 256, 'decoder_conv0', depth_activation=False, point_activation=True, epsilon=1e-5)
        seg = self._SepConv_BN(seg, 256, 'decoder_conv1', depth_activation=False, point_activation=True, epsilon=1e-5)
        seg = self._SepConv_BN(seg, 256, 'decoder_conv2', depth_activation=False, point_activation=True, epsilon=1e-5)
        seg = self._SepConv_BN(seg, 256, 'decoder_conv3', depth_activation=False, point_activation=True, epsilon=1e-5)
        seg = self._SepConv_BN(seg, 256, 'decoder_conv4', depth_activation=False, point_activation=True, epsilon=1e-5)
        seg = self._SepConv_BN(seg, self.num_classes, 'seg_depth_point_3', epsilon=1e-5)

        return center, reg, seg

    def _build_graph(self):

        s_4, s_8, s_16, s_32 = self._build_backbone(self.images)
        s_8 = self._fusion_feature(s_4, s_8, s_16, s_32)
        keypoints, preg, fpn = self._detect_head(s_8)

        if self.mode == 'train':
            center_loss = CentLoss(self.batch_size, self.num_classes, self.loss_decay, self.stride)\
                ([keypoints, preg, fpn, self.ground_truth])
            inputs = [self.images, self.ground_truth]
            outputs = [keypoints, preg, fpn, center_loss]
        else:
            # [h, w, 2]
            pshape = [tf.shape(keypoints)[1], tf.shape(keypoints)[2]]
            h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
            w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
            [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
            meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)
            meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
            center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)

            # [1, y, x, ?]
            pic_keypoints = tf.sigmoid(keypoints)
            pic_seg = tf.sigmoid(fpn)

            # [y, x, 1]
            category = tf.expand_dims(tf.argmax(pic_keypoints, axis=-1, output_type=tf.int32)[0], axis=-1)

            # [y, x, 1 + 2(y, x) + 1(index_of_class)=4]
            meshgrid_xyz = tf.concat([tf.zeros_like(category), tf.cast(center, tf.int32), category], axis=-1)

            # [1, y, x, 1]
            pic_keypoints = tf.gather_nd(pic_keypoints, meshgrid_xyz)
            pic_keypoints = tf.expand_dims(pic_keypoints, axis=0)
            pic_keypoints = tf.expand_dims(pic_keypoints, axis=-1)

            # 3*3 to be peak value
            keypoints_peak = self._max_pooling(pic_keypoints, 3, 1)
            # mask for each peak_point in each 3*3 area, [1, y, x, 1] (0,1)
            keypoints_mask = tf.cast(tf.equal(pic_keypoints, keypoints_peak), tf.float32)
            # [1, y, x, 1] (true, false)
            pic_keypoints = pic_keypoints * keypoints_mask
            # [y*x]
            scores = tf.reshape(pic_keypoints, [-1])
            # [y*x]
            class_id = tf.reshape(category, [-1])
            # [(y* x), 2]
            grid_yx = tf.reshape(center, [-1, 2])
            # [(y*x), 4]
            bbox_lrtb = tf.reshape(preg, [-1, 4])

            # # # TODO: ATTENTION, order are lrtb in prediction, but tlbr in ground_truth
            # score_mask = scores > self.score_threshold
            # # scores = scores - self.score_threshold
            # # score_mask = tf.cast(Activation('relu')(scores), tf.bool)
            #
            # select_scores = tf.boolean_mask(scores, score_mask)
            # select_class_id = tf.boolean_mask(class_id, score_mask)
            # select_center = tf.boolean_mask(grid_yx, score_mask)
            # select_lrtb = tf.boolean_mask(bbox_lrtb, score_mask)
            # # [num, 4(y1, x1, y2, x2)]
            # select_bbox = tf.concat([select_center - select_lrtb[..., 0:2], select_center + select_lrtb[..., 2:4]], axis=-1)

            # TODO: TF MNS
            bbox = tf.concat([grid_yx - bbox_lrtb[..., 0:2], grid_yx + bbox_lrtb[..., 2:4]],
                                    axis=-1)
            select_indices = tf.image.non_max_suppression(bbox, scores, self.top_k_results_output,
                                                          self.nms_threshold, score_threshold=self.score_threshold)
            # [num_select, ?]
            select_scores = tf.gather(scores, select_indices)
            select_center = tf.gather(grid_yx, select_indices)
            select_class_id = tf.gather(class_id, select_indices)
            select_bbox = tf.gather(bbox, select_indices)

            pic_seg = tf.cast(pic_seg > self.seg_threshold, tf.float32)
            select_scores = tf.expand_dims(select_scores, axis=0)
            select_center = tf.expand_dims(select_center, axis=0)
            select_class_id = tf.expand_dims(select_class_id, axis=0)
            select_bbox = tf.expand_dims(select_bbox, axis=0)

            # for post_processing outputs
            outputs = [select_center, select_scores, select_bbox, select_class_id, pic_seg, preg]
            inputs = [self.images]
        self.CenterNetModel = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(self):
        """Gets the model ready for training. Adds l
        osses including regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Add L2 Regularization
        reg_losses = self.l2_decay * tf.add_n([tf.nn.l2_loss(var) for var in self.CenterNetModel.trainable_weights])
        self.CenterNetModel.add_loss(lambda: tf.reduce_sum(reg_losses))

        # Optimizer object
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

        self.CenterNetModel.compile(optimizer=optimizer)

    def train_epochs(self, dataset, valset, config, epochs=50):
        self.compile()
        # iter_data = dataset.generator(config.BATCH_SIZE, config.STEPS_PER_EPOCH)
        # val_generator = valset.generator(config.BATCH_SIZE, config.VALIDATION_STEPS)

        epochRec = EpochRecord(self.name)
        callbacks = [
            epochRec,
            tf.keras.callbacks.ProgbarLogger(),
            # tf.keras.callbacks.ReduceLROnPlateau(moniter='val_loss', factor=0.1, patience=2, mode='min', min_lr=1e-7),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True,
                                           write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path + "/weights.{epoch:03d}-{loss:.2f}.hdf5", verbose=0, save_weights_only=True)
        ]
        step = int(config.PIC_NUM / config.BATCH_SIZE)
        print("=====ready for model.fit_generator======")
        self.CenterNetModel.fit_generator(
            dataset,
            initial_epoch=self.pro_epoch,
            epochs=epochs,
            max_queue_size=4,
            workers=1,
            steps_per_epoch=step,
            use_multiprocessing=False,
            # validation_data=val_generator,
            # validation_steps=self.config.VALIDATION_STEPS,
            # validation_freq=1,
            callbacks=callbacks
        )

    def test_one_image(self, images, show=False):
        self.is_training = False
        image, window, scale, padding, crop = resize_image(
            images,
            min_dim=self.image_size,
            min_scale=0,
            max_dim=self.image_size,
            mode="square")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = np.reshape(mean, [1, 1, 3])
        std = np.reshape(std, [1, 1, 3])
        image = (image / 255. - mean) / std
        image = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        # self.CenterNetModel.save('./SAVE')
        # tf.saved_model.save(self.CenterNetModel, "./SAVE")
        # tf.keras.experimental.export_saved_model(self.CenterNetModel, "./SAVE")

        # self.CenterNetModel.save('my_model.h5')
        #
        # converter = tf.lite.TFLiteConverter.from_keras_model('my_model.h5')
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        # tflite_model = converter.convert()
        # open("converted_model.tflite", "wb").write(tflite_model)

        # float16 quantilize
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.CenterNetModel)
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.float16]
        # tflite_quant_model = converter.convert()
        # open("converted_model.tflite", "wb").write(tflite_quant_model)

        pred = self.CenterNetModel.predict(
            image,
            batch_size=1,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )
        return pred

    def load_weight(self, epoch):
        # latest = tf.train.latest_checkpoint(self.checkpoint_path)
        epoch = str(epoch).zfill(3)
        latest = ""
        for filename in os.listdir(self.checkpoint_path):
            root, ext = os.path.splitext(filename)
            if root.startswith('weights.' + epoch) and ext == '.hdf5':
                latest = filename
                break
        self.CenterNetModel.load_weights("./" + self.checkpoint_path + "/" + latest, by_name=True)
        print('load weight', latest, 'successfully')

    def load_pretrained_weight(self, path):
        self.pretrained_saver.restore(self.sess, path)
        print('load pretrained weight', path, 'successfully')

    def _bn(self, bottom):
        bn = tf.keras.layers.BatchNormalization()(bottom)
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _conv_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        if activation is not None:
            return activation(conv)
        else:
            return conv

    def _dconv_bn_activation(self, bottom, filters, kernel_size, strides, activation=h_swish):
        conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )(bottom)
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        a = tf.keras.layers.MaxPool2D(
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )(bottom)
        return a

    def _SepConv_BN(self, x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False,
                    point_activation=False, epsilon=1e-3):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
            Implements right "same" padding for even kernel sizes
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & poinwise convs
                epsilon: epsilon to use in BN layer
        """

        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)  # without padding around kernel
            pad_total = kernel_size_effective - 1  # padding for feature map
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if depth_activation:
            x = Activation('relu')(x)
        # # stride != 1 is incompatible with dilation_rate != 1
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=prefix + '_pointwise')(x)
        if point_activation:
            x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
            x = Activation('relu')(x)

        return x
