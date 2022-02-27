import cv2
import matplotlib.pyplot as plt
video = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = video.read()
    rect = [250, 250]
    cv2.rectangle(frame, (150, 40),
                  (150+rect[0], 40+rect[1]), (0, 255, 0), 2)
    img_cut = frame[40:40+rect[1], 150:150+rect[0]]

    cv2.imshow('video', img_cut)
    if cv2.waitKey(30) == ord('q'):
        break
    elif cv2.waitKey(30) == ord('z'):
        print(i)
        img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
        plt.imsave(f'./tmp_dataset/lowerHead/{i}.jpg', img_cut)
        # cv2.imwrite(f'./data/正视/{i}.jpg', img_cut)

        i += 1
        if i == 150:
            break
