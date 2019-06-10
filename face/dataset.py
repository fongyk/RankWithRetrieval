import os
import shutil

def buildTrainingFace():
    face_path = '/data4/fong/face/CASIA_WebFace'
    person_folder = os.listdir(face_path)
    person_folder.sort()
    ftrain = open('train.txt', 'w')
    for person in person_folder:
        source_path = face_path + '/' + person + '\n'
        ftrain.write(source_path)

if __name__ == '__main__':
    buildTrainingFace()
    print 'finished!'
