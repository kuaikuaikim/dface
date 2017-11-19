import cv2
from core.detect import create_mtcnn_net, MtcnnDetector
import core.vision as vision




if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch_5best.pt", r_model_path="./model_store/rnet_epoch_1.pt", o_model_path="./model_store/onet_epoch_7bbest.pt", use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./test.jpg")
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align

    vision.vis_face(img2,bboxs,landmarks)