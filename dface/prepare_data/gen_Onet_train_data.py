import argparse

import cv2
import numpy as np
from dface.core.detect import MtcnnDetector,create_mtcnn_net
from dface.core.imagedb import ImageDB
from dface.core.image_reader import TestImageLoader
import time
import os
import cPickle
from dface.core.utils import convert_to_square,IoU
import dface.config as config
import dface.core.vision as vision

def gen_onet_data(data_dir, anno_file, pnet_model_file, rnet_model_file, prefix_path='', use_cuda=True, vis=False):


    pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file, r_model_path=rnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)

    imagedb = ImageDB(anno_file,mode="test",prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb,1,False)

    all_boxes = list()
    batch_idx = 0

    for databatch in image_reader:
        if batch_idx % 100 == 0:
            print("%d images done" % batch_idx)
        im = databatch

        t = time.time()

        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)

        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        t1 = time.time() - t
        t = time.time()
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = config.MODEL_STORE_DIR

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


    gen_onet_sample_data(data_dir,anno_file,save_file,prefix_path)






def gen_onet_sample_data(data_dir,anno_file,det_boxs_file,prefix):

    neg_save_dir = os.path.join(data_dir, "48/negative")
    pos_save_dir = os.path.join(data_dir, "48/positive")
    part_save_dir = os.path.join(data_dir, "48/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix,annotation[0])

        boxes = map(float, annotation[1:])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)


    save_path = config.ANNO_STORE_DIR
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    det_handle = open(det_boxs_file, 'r')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()



def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dface_traindata_store', dest='traindata_store', help='dface train data temporary folder,include 12,24,48/postive,negative,part,landmark',
                        default='../data/wider/', type=str)
    parser.add_argument('--anno_file', dest='annotation_file', help='wider face original annotation file',
                        default=os.path.join(config.ANNO_STORE_DIR,"wider_origin_anno.txt"), type=str)
    parser.add_argument('--pmodel_file', dest='pnet_model_file', help='PNet model file path',
                        default='/idata/workspace/dface/model_store/pnet_epoch.pt', type=str)
    parser.add_argument('--rmodel_file', dest='rnet_model_file', help='RNet model file path',
                        default='/idata/workspace/dface/model_store/rnet_epoch.pt', type=str)
    parser.add_argument('--gpu', dest='use_cuda', help='with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='prefix_path', help='annotation file image prefix root path',
                        default='', type=str)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    gen_onet_data(args.traindata_store, args.annotation_file, args.pnet_model_file, args.rnet_model_file, args.prefix_path, args.use_cuda)



