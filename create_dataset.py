import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
        return True
    except:
        return False


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, Type, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        try:
            imagePath = imagePathList[i]
            #imagePath = "./" + Type + "/" + imagePath
            label = labelList[i]
            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue
            with open(imagePath, 'r') as f:
                imageBin = f.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label
            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[i])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
        except:
            print("Warning!")
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def makeDataset(inputPath, Type):
    image_list = []
    label_list = []
    with open(inputPath, 'r') as f:
        lines = f.readlines()
        from random import shuffle
        shuffle(lines)
        for line in lines:
            words = line.strip("\n").split('\t')
            image_file = words[0]
            label = words[1]
            image_list.append(image_file)
            label_list.append(label)
    return image_list, label_list

if __name__ == '__main__':
    os.system("rm -rf train_lmdb; rm -rf ./test_lmdb")
    Type = "train"
    image_list, label_list = makeDataset("./data3/train.txt", Type)
    createDataset("./train_lmdb", image_list, label_list, Type, checkValid = True)

    Type = "test"
    image_list, label_list = makeDataset("./data3/test.txt", Type)
    createDataset("./test_lmdb", image_list, label_list, Type, checkValid = True)
