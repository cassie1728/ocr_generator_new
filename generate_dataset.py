#encoding=utf-8
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool
import time
import random
from random import randint

from img_utils import add_effects_to_image, paste_text, paste_vertical_text

reload(sys)
sys.setdefaultencoding('utf-8')

DATA_PATH = "./data4/"
CORPUS_FILE_PATH = "./ocr_words.txt"
BG_IMAGES_PATH = "/data/zhangjiaxuan/ocr_dataset_generator/paopao-new/no_text_images/"
FONT_FILE_PATH = "./fonts_66/"
LABEL_FILE_PATH = "./words_list_paopao.txt"

NUM_PER_PROCESS = 50
NUM_PER_IMAGE = 10
LABEL_MAX_LEN = 73

DEBUG = False

# get background image from background image list 
# or pure color image.
# maybe we could add some noise to the pure color
# because it's too pure!
def get_bg_img(bg_img_list):
    rd = randint(0, 5)
    if rd == 0:
        bg_color = get_random_color()
        bg_img = Image.new("RGB", (500, 500), bg_color) #1/5彩色背景
    elif rd == 1:
        bg_img = Image.new("RGB", (500, 500), "white") #1/5白色背景

        return bg_img

    bg_img_path = get_random_element(bg_img_list)
    try:
        bg_img = Image.open(bg_img_path).convert("RGB")
        width, height = bg_img.size
    except Exception as e:
        print "Error in loading background image:", bg_img_path
        print str(e)
        return None

    return bg_img

def get_file_list(path):
    file_list = []
    for root, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            full_path = os.path.join(root, file_name)
            file_list.append(full_path)

    return file_list

def generate_corpus(charset_file):
    char_list = []
    with open(charset_file, 'r') as f:
        for line in f.readlines():
            char = line.strip("\n\n").strip("\n").decode("utf-8")
            char_list.append(char)

    char_list.append(" ")
    charset_len = len(char_list)

    corpus_list = []

    for i in range(charset_len):
        corpus = ""
        if randint(0,10) == 0:
            corpus_len = randint(10, 20)
        else:
            corpus_len = randint(1, 10)
        for j in range(corpus_len):
            corpus += char_list[randint(0, charset_len-1)]

        corpus_list.append(corpus)

    return corpus_list

def get_corpus_list(font_file):
    GENERATE_CORPUS = True
    if GENERATE_CORPUS == True:
        charset_file = "./charsets_66/" + os.path.basename(font_file)[:-4] + ".txt"
        if not os.path.exists(charset_file):
            return []
        print "Generate corpus()"
        corpus_list = generate_corpus(charset_file)
        print "Generate_corpus() finished!"
    else:
        corpus_list = []
        with open(CORPUS_FILE_PATH, 'r') as f:
            for line in f.readlines():
                line = line.strip("\r\n").strip("\n")
                corpus_list.append(line.decode("utf-8"))

    return corpus_list

def get_bg_images():
    bg_img_list = []

    for root, dir_names, file_names in os.walk(BG_IMAGES_PATH):
        for dir_name in dir_names:
            dir_file_list = get_file_list(os.path.join(root, dir_name))
            for bg_img in dir_file_list:
                bg_img_list.append(bg_img)

    return bg_img_list

def get_random_color():
    shift = 255
    r = randint(255-shift, 255)
    g = randint(255-shift, 255)
    b = randint(255-shift, 255)

    return (r,g,b)

def get_random_element(src_list):
    return src_list[randint(0, len(src_list) - 1)]

def read_label_dict(dict_file):
    label_dict = {}
    with open(dict_file, 'r') as f:
        lines = f.readlines()
        index = 1
        for line in lines:
            line = line.strip("\r\n").strip("\n").decode("utf-8")
            label_dict[line] = str(index)
            index = index + 1

    return label_dict

def convert_corpus_to_label(corpus):
    label_dict = read_label_dict(LABEL_FILE_PATH)

    corpus = corpus
    if len(corpus) > LABEL_MAX_LEN:
        print "The len of corpus is bigger than LABEL_MAX_LEN:", corpus
        return None
    label = ""
    for char in corpus:
        if char not in label_dict:
            print "Error:", char, "not in label file!"
            continue
        label += label_dict[char] + " "

    if len(label.replace(" ", "")) == 0:
        return None

    return label

def get_font_list():
    return get_file_list(FONT_FILE_PATH)

def process(path_index):
    """
    1. Saved directory initialization;
    2. Get coupus list and background image list;
    3. Get font file(In the process(), only use one font file);
    4. Recursively generate images:
        4.1 Every NUM_PER_IMAGE use a background image;
        4.2 Get corpus, font color and font size;
        4.3 Paste text onto the background image;
        4.4 Add different effects to printed image;
        4.5 Append saved image path to return list;
    5. Return image path list;
    """
    start_time = time.time()
    print "process %d is start." %(path_index)
    save_path = "./data4/%03d/" %(path_index)
    print save_path
    if not os.path.exists(save_path):
        os.system("mkdir " + save_path)

    bg_img_list = get_bg_images()
    font_list   = get_font_list()
    font_file   = font_list[path_index%(len(font_list))]
    print font_file
    corpus_list = get_corpus_list(font_file)
    if len(corpus_list) == 0:
        return ([], [], [])

    saved_img_path_list = []
    saved_corpus_list = []
    saved_label_list = []

    for i in range(NUM_PER_PROCESS):
        if i % NUM_PER_IMAGE == 0:
            bg_img = get_bg_img(bg_img_list)
            if bg_img is None:
                continue

        corpus = get_random_element(corpus_list)

        font_color = get_random_color()
        font_size = randint(25, 45) # simsun.ttf字体，字号小于18时，打印的字符是乱码

        PRINT_HORIZONAL_TEXT = True
        if PRINT_HORIZONAL_TEXT == True:
            printed_img, text_rect = paste_text(bg_img, corpus, font_color, font_file, font_size)
        else:
            printed_img, text_rect = paste_vertical_text(bg_img, corpus, font_color, font_file, font_size)

        if printed_img == None:
            continue

        dst_img, printed_img = add_effects_to_image(printed_img,text_rect, PRINT_HORIZONAL_TEXT)
        if dst_img == None:
            continue

        label = convert_corpus_to_label(corpus)
        if label == None:
            continue

        # path_index-font_file-font_size-image_index.jpg
        saved_img_path = "./data4/%03d/%03d_%s_%d_%05d.jpg" %(path_index, path_index, os.path.basename(font_file)[:-4], font_size, i)

        if PRINT_HORIZONAL_TEXT == False:
            dst_img = dst_img.transpose(Image.ROTATE_90)
        dst_img.save(saved_img_path)
        if DEBUG == True:   # save the whole printed image for debug
            debug_img_path = "./data4/%03d/%03d_%s_%d_%05d_src.jpg" %(path_index, path_index, os.path.basename(font_file)[:-4], font_size, i)
            printed_img.save(debug_img_path)

        saved_img_path_list.append(saved_img_path)
        saved_corpus_list.append(corpus)
        saved_label_list.append(label)

    print "Process %d is finished! %0.0fs" %(path_index, time.time()-start_time)

    return (saved_img_path_list, saved_corpus_list, saved_label_list)

def save_label(img_path_list, corpus_list, label_list):
    with open("./data4/test.txt", 'a+') as f_test, open("./data4/test_corpus.txt", 'a+') as f_test_corpus, open("./data4/train.txt", 'a+') as f_train, open("./data4/train_corpus.txt", "a+") as f_train_corpus:
        for i, img_path in enumerate(img_path_list):
            if i < len(img_path_list)/10:
                f = f_test
                f_corpus = f_test_corpus
            else:
                f = f_train
                f_corpus = f_train_corpus
            f.write(img_path + "\t" + label_list[i] + "\n")
            f_corpus.write(img_path + "\t" + corpus_list[i] + "\n")

def save_labels_to_file(return_list):
    if not isinstance(return_list, list):
        img_path_list, corpus_list, label_list = return_list

        save_label(img_path_list, corpus_list, label_list)
    else:
        for one_list in return_list:
            save_labels_to_file(one_list)

def dir_init():
    if os.path.exists(DATA_PATH):
        os.system("rm -r " + DATA_PATH)

    os.system("mkdir " + DATA_PATH)

if __name__ == "__main__":
    """
    1. Initialize directories: remove old dirs and create empty dirs;
    2. generate train images and test images;
    """
    start_time = time.time()
    dir_init()
    
    font_file_len = len(get_font_list())
    print font_file_len
    MULTI_PROCESSING = True
    if MULTI_PROCESSING:
        process_no = font_file_len
        index_list = range(0, process_no)

        pool_no = 10
        pool = Pool(pool_no)
        return_list = pool.map_async(process,index_list)
        pool.close()
        pool.join()

        return_list = return_list.get()
    else:
        return_list = []
        for i in range(0, font_file_len):
            one_list = process(i)
            return_list.append(one_list)

    save_labels_to_file(return_list)

    print("Finished to generate dataset. %0.0fs" %(time.time()-start_time))
