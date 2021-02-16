import dlib
import cv2
import numpy as np
from collections import OrderedDict

FACIAL_LANDMARKS_68_IDXS = OrderedDict([("mouth", (48, 68)), ("right_eyebrow", (17, 22)), ("left_eyebrow", (22, 27)),
                                        ("right_eye", (36, 42)), ("left_eye", (42, 48)), ("nose", (27, 36)),
                                        ("jaw", (0, 17))])


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), (158, 163, 32), (163, 38, 32),
                  (180, 42, 220)]
    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        if name == "jaw":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread('./images/liudehua2.jpg')
(h, w) = image.shape[:2]
width = 500
r = width / float(w)
dim = (width, int(h * r))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

        roi = image[y:y + h, x:x + w]
        (h, w) = roi.shape[:2]
        width = 250
        r = width / float(w)
        dim = (width, int(h * r))
        roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)

    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
