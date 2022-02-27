import numpy as np


def erase_repeated_rect(rect1s, rect2s):
    print(len(rect1s), len(rect2s))
    final_center_rects = []
    rect1_center = []
    rect2_center = []
    for rect1 in rect1s:
        rect1_center.append(caculate_center(rect1))
    for rect2 in rect2s:
        rect2_center.append(caculate_center(rect2))

    for rect_ct1 in rect1_center:
        # if rect_ct1 in final_center_rects:
        #     continue
        for rect_ct2 in rect2_center:
            print('is running')

            if center_loss(rect_ct1, rect_ct2, 20):
                if rect_ct2 not in final_center_rects:
                    final_center_rects.append(rect_ct2)

                if rect_ct1 not in final_center_rects:
                    final_center_rects.append(rect_ct1)
            else:
                if rect_ct1 not in final_center_rects or rect_ct2 not in final_center_rects:

                    final_center_rects.append(rect_ct1)
    print(len(final_center_rects))
    return final_center_rects


def caculate_center(rect):
    return ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)


def center_loss(rect_ct1, rect_ct2, loss):
    if np.sqrt((rect_ct2[0]-rect_ct1[0])**2+(rect_ct2[1]-rect_ct2[1])**2) < loss:
        return False
    return True
