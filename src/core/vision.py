from matplotlib.patches import Circle


def vis_two(im_array, dets1, dets2, thresh=0.9):
    """Visualize detection results before and after calibration

    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results before calibration
    dets2: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results after calibration
    thresh: float
        boxes with scores > thresh will be drawn in red otherwise yellow

    Returns:
    -------
    """
    import matplotlib.pyplot as plt
    import random

    figure = plt.figure()
    plt.subplot(121)
    plt.imshow(im_array)
    color = 'yellow'

    for i in range(dets1.shape[0]):
        bbox = dets1[i, :4]
        landmarks = dets1[i, 5:]
        score = dets1[i, 4]
        if score > thresh:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)
            landmarks = landmarks.reshape((5,2))
            for j in range(5):
                plt.scatter(landmarks[j,0],landmarks[j,1],c='yellow',linewidths=0.1, marker='x', s=5)


            # plt.gca().text(bbox[0], bbox[1] - 2,
            #                '{:.3f}'.format(score),
            #                bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
        # else:
        #     rect = plt.Rectangle((bbox[0], bbox[1]),
        #                          bbox[2] - bbox[0],
        #                          bbox[3] - bbox[1], fill=False,
        #                          edgecolor=color, linewidth=0.5)
        #     plt.gca().add_patch(rect)

    plt.subplot(122)
    plt.imshow(im_array)
    color = 'yellow'

    for i in range(dets2.shape[0]):
        bbox = dets2[i, :4]
        landmarks = dets1[i, 5:]
        score = dets2[i, 4]
        if score > thresh:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)

            landmarks = landmarks.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks[j, 0], landmarks[j, 1], c='yellow',linewidths=0.1, marker='x', s=5)

            # plt.gca().text(bbox[0], bbox[1] - 2,
            #                '{:.3f}'.format(score),
            #                bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
        # else:
        #     rect = plt.Rectangle((bbox[0], bbox[1]),
        #                          bbox[2] - bbox[0],
        #                          bbox[3] - bbox[1], fill=False,
        #                          edgecolor=color, linewidth=0.5)
        #     plt.gca().add_patch(rect)
    plt.show()


def vis_face(im_array, dets, landmarks=None):
    """Visualize detection results before and after calibration

    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results before calibration
    dets2: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results after calibration
    thresh: float
        boxes with scores > thresh will be drawn in red otherwise yellow

    Returns:
    -------
    """
    import matplotlib.pyplot as plt
    import random
    import pylab

    figure = pylab.figure()
    # plt.subplot(121)
    pylab.imshow(im_array)
    figure.suptitle('DFace Detector', fontsize=20)



    for i in range(dets.shape[0]):
        bbox = dets[i, :4]

        rect = pylab.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='yellow', linewidth=0.9)
        pylab.gca().add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                # pylab.scatter(landmarks_one[j, 0], landmarks_one[j, 1], c='yellow', linewidths=0.1, marker='x', s=5)

                cir1 = Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]), radius=2, alpha=0.4, color="red")
                pylab.gca().add_patch(cir1)
                # plt.gca().text(bbox[0], bbox[1] - 2,
                #                '{:.3f}'.format(score),
                #                bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
                # else:
                #     rect = plt.Rectangle((bbox[0], bbox[1]),
                #                          bbox[2] - bbox[0],
                #                          bbox[3] - bbox[1], fill=False,
                #                          edgecolor=color, linewidth=0.5)
                #     plt.gca().add_patch(rect)

        pylab.show()