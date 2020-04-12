#!/usr/bin/env python3
import cv2
import numpy as np
from custom.LucasKanade import LucasKanade

FRAME1="../data/DragonBaby/img/0001.jpg"
FRAME2="../data/DragonBaby/img/0002.jpg"
TEMPLATE_BBOX=[160,83,56,65]

if __name__ == "__main__":
    template = cv2.imread(FRAME1,0)
    frame2 = cv2.imread(FRAME2,0)

    # cv2.imshow("bb",frame2)
    # cv2.waitKey(0)

    lk = LucasKanade(template,TEMPLATE_BBOX)
    lk.estimate(frame2)


