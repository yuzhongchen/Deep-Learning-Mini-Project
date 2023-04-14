import os
import cv2

path = './cifar10/train/'
lists = os.listdir(path)
lists.sort()

for list in lists:
    list_name = os.listdir('%s%s' % (path, list))
    if not os.path.exists('%s%s_1' % (path, list)):
        os.mkdir('%s%s_1' % (path, list))
    for name in list_name:
        name = name[:-4]
        img = cv2.imread('%s%s/%s.png' % (path, list, name))
        img_0 = cv2.flip(img, 0)
        img_1 = cv2.flip(img, 1)
        img_2 = cv2.flip(img, -1)
        cv2.imwrite('%s%s_1/%s_0.png' % (path, list, name), img_0)
        cv2.imwrite('%s%s_1/%s_1.png' % (path, list, name), img_1)
        cv2.imwrite('%s%s_1/%s_2.png' % (path, list, name), img_2)
