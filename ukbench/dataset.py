import os
import shutil
import random

def buildUkb():
    img_path = '/path/to/images'
    img_file = os.listdir(img_path)
    ftrain = open('train.txt', 'w')
    for img in img_file:
        source_img = img_path + '/' + img
        img_id = int(img[-9:-4])
        group_id = img_id / 4
        if img_id < 6200:
            des_path = '/path/to/{:04d}'.format(group_id)
            if not os.path.exists(des_path):
                os.mkdir(des_path)
            des_img = des_path + '/' + img
            ftrain.write(des_img+' '+str(group_id)+'\n')
            # shutil.move(source_img, des_img)
            if not os.path.exists(des_img):
                shutil.copy(source_img, des_img)
        else:
            # des_path = '/path/to/test/{:04d}'.format(group_id)
            # if not os.path.exists(des_path):
                # os.mkdir(des_path)
            des_path = '/path/to/test'
            des_img = des_path + '/' + img
            if not os.path.exists(des_img):
                shutil.copy(source_img, des_img)
    ftrain.close()

def buildAnchors():

    img_path = '/path/to/images'
    img_file = os.listdir(img_path)
    img_file.sort()
    group = 6200/4
    for idx in range(group):
        img_id = idx * 4 + random.randint(0, 3)
        source_img = img_path + '/' + img_file[img_id]
        des_path = '/path/to/anchors/{:04d}'.format(idx)
        if not os.path.exists(des_path):
            os.mkdir(des_path)
        des_img = des_path + '/'  + img_file[img_id]
        if not os.path.exists(des_img):
            shutil.copy(source_img, des_img)

if __name__ == '__main__':
    buildUkb()
    buildAnchors()
    print 'finished!'
