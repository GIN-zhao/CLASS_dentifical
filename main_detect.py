import argparse
import sys
import os
import cv2
from sklearn import utils
import tensorflow as tf
import numpy as np
from mtcnn import mtcnn
from caculate_rect import erase_repeated_rect
from train_CNN import read_data
# import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# parser = argparse.ArgumentParser(
#     description='convert model')

# parser.add_argument('--net_type', default="slim", type=str,
#                     help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
# parser.add_argument('--img_path', default='./imgs/test_input.jpg', type=str,
#                     help='Image path for inference')
# args = parser.parse_args()


def detect(img):
    # if args.net_type == 'slim':
    model_path = "./export_models/RFB/"
    # elif args.net_type == 'RFB':
    #     model_path = "export_models/RFB/"
    # else:
    #     print("The net type is wrong!")
    #     sys.exit(1)

    actions = ['straight', 'slide_face', 'uphead', 'lowerhead']
    online = ['straight']
    model = tf.keras.models.load_model(model_path)
    # model2 = mtcnn()
    threshold = [0.5, 0.6, 0.7]
    detect = tf.keras.models.load_model('./export_models/CNN.h5')
    # video = cv2.VideoCapture(0)
    # while True:
    # try:
    #         ret, img = video.read()
    # print(img)
    h, w, _ = img.shape
    # temp_img = img.copy()
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0
    # result=[background,face,x1,y1,x2,y2]
    results = model.predict(np.expand_dims(img_resize, axis=0))
    # rectangles = model2.detectFace(temp_img, threshold)
    rect1s = []
    rect2s = []
    # for rectangle in rectangles:
    #     rect_temp = [rectangle[0], rectangle[1],
    #                  rectangle[2], rectangle[3]]
    #     map(int, rect_temp)
    #     rect1s.append(rect_temp)
    #     cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])), (int(
    #         rectangle[2]), int(rectangle[3])), (0, 0, 255), 2)
    for result in results:
        start_x = int(result[-4] * w)-40
        start_y = int(result[-3] * h)-40
        end_x = int(result[-2] * w)+40
        end_y = int(result[-1] * h)+40
    #     rect_temp = [start_x, start_y, end_x, end_y]
    #     rect2s.append(rect_temp)
    # final_rect=utils.NMS((rect1s,rect2s),0.8)
    # for start_x,
        # cv2.rectangle(img, (start_x, start_y),
        #               (end_x, end_y), (0, 255, 0), 2)
    # final_center = erase_repeated_rect(rect1s, rect2s)
    # for center in final_center:
    #     start_x = int(center[0]-125)
    #     start_y = int(center[1]-125)
    #     end_x = int(center[0] + 125)
    #     end_y = int(center[1]+125)
    #     print(start_x, start_y, end_x, end_y)
        cv2.rectangle(img, (start_x-30, start_y-30),
                      (end_x+30, end_y+30), (0, 255, 0), 2)
        head_action_img = img[start_y:end_y, start_x:end_x]
        head_action_img = cv2.resize(head_action_img, dsize=(250, 250))
        head_action_img = np.expand_dims(head_action_img, axis=0)
        # print(head_action_img.shape)
        pred = detect.predict(head_action_img)
        # print(pred)
        index = np.argmax(pred)
        if actions[index] in online:
            cv2.putText(img, 'online',
                        (start_x, start_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        else:
            cv2.putText(img, 'leaving',
                        (start_x, start_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    # cv2.imwrite(f'imgs/test_output_{args.net_type}.jpg', img)
    img = cv2.resize(img, (640, 480))
    return img
    # cv2.imshow('predict', img)
    # if cv2.waitKey(30) == ord('q'):
    #     break
    # return img
    # cv2.imshow('test_output', img)
    # cv2.waitKey(0
    # cv2.imwrite(
    #     'F:\\deskTop\\homework\\MachineLearning\\face_count\\myAction_FACE\\imgs\test_output.png', img)
    # except:
    #     pass


# if __name__ == '__main__':
#     test_img = cv2.imread('./imgs/test.png')
#     print(test_img.shape)
#     detect(test_img)
#     main()
    # x_train, x_test, y_train, y_test = read_data()
