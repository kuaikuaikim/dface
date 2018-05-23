import cv2
from dface.core.detect import create_mtcnn_net, MtcnnDetector
import dface.core.vision as vision




if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch.pt", r_model_path="./model_store/rnet_epoch.pt", o_model_path="./model_store/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./test.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align

    vision.vis_face(img_bg,bboxs,landmarks)
