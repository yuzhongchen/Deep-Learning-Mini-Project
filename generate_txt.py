import os
import random

path = './cifar10/train/'
lists = os.listdir(path)
lists.sort()
i = 0
j = 0
label_list = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
# for list in lists:
#     list_name = os.listdir('%s%s' % (path, list))
#     for name in list_name:
#         temp = random.random()
#         if temp <=0.8:
#             listText = open('./cifar10/train_200000.txt', 'a+')
#             label = str(label_list[i])
#             filename = path + list + '/' + name + '\t' + label + '\n'
#             listText.write(filename)
#             listText.close()
#         else:
#             listText = open('./cifar10/val.txt', 'a+')
#             label = str(label_list[i])
#             filename = path + list + '/' + name + '\t' + label + '\n'
#             listText.write(filename)
#             listText.close()
#     i += 1
# label_list = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
for list in lists:
    if j % 2 == 0:
        list_name = os.listdir('%s%s' % (path, list))
        for name in list_name:
            listText = open('cifar10/train.txt', 'a+')
            label = str(label_list[j])
            filename = path + list + '/' + name + '\t' + label + '\n'
            listText.write(filename)
            listText.close()
    i += 1
    j += 1