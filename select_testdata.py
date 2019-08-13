import os
import shutil

tarDir = './data2_test'
if not os.path.exists(tarDir):
    os.makedirs(tarDir)
else:
    shutil.rmtree(tarDir)
    os.makedirs(tarDir)

#label_path=tarDir+'/'+'test_label.txt'
#if os.path.exists(label_path):
#    os.remove(label_path)

with open ("./data2/test_corpus.txt",'r',encoding='utf-8') as f,open(tarDir+'/'+'test_label.txt','a',encoding='utf-8') as f1:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        words = line.split("\t")
        image_path = words[0]
        image_name = image_path[11:]
        if not os.path.exists(image_path[:11]):
            continue
        else:
            shutil.copyfile(image_path, tarDir+'/'+image_name)
            f1.write(image_name+'\t'+words[1]+'\n')
