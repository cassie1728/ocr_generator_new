#encoding=utf-8
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import random
from random import randint
import numpy as np
import math

# convert rect style form x,y,w,h to x1,y1,x2,y2,x3,y3,x4,y4
def convert_rect_style(bbox):
    x,y,w,h = bbox
    return ((x, y, x+w, y, x+w, y+h, x, y+h))

def get_transform_bbox(a,b,c,d,e,f, bbox):
    bbox = np.array([[bbox[0], bbox[2], bbox[4], bbox[6]],
                     [bbox[1], bbox[3], bbox[5], bbox[7]],
                     [1, 1, 1, 1]])
    M = np.array([[a,b,c],
         [d,e,f],
         [0,0,1]])
    # M is the inverse of the orign transform matrix(because of PIL's function transform)
    # so, we need to inverse M to get the origin transform matrix
    inv_M = np.linalg.inv(M)

    bbox = inv_M.dot(bbox)
    new_bbox = [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1], bbox[0][2], bbox[1][2], bbox[0][3], bbox[1][3]]
    new_bbox = map(int, new_bbox)

    return new_bbox

"""
Apply affine transform to the image according to the parameters,
return the transformed image and transformed bbox.
"""
def affine_transform(im, sx, sy, angle, bbox):
    img_orig = im
    im = Image.new("RGB", img_orig.size, (255, 255, 255))
    im.paste(img_orig)

    w, h = im.size

    angle = math.radians(-angle)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    scaled_w, scaled_h = w*sx, h*sy

    new_w = int(math.ceil(math.fabs(cos_theta*scaled_w) + math.fabs(sin_theta*scaled_h)))
    new_h = int(math.ceil(math.fabs(sin_theta*scaled_w) + math.fabs(cos_theta*scaled_h)))

    cx = w/2.
    cy = h/2.
    tx = new_w/2.
    ty = new_h/2.

    a = cos_theta/sx
    b = sin_theta/sx
    c = cx-tx*a-ty*b
    d = -sin_theta/sy
    e = cos_theta/sy
    f = cy-tx*d-ty*e

    new_bbox = get_transform_bbox(a,b,c,d,e,f,bbox)

    return new_bbox, im.transform((new_w, new_h), Image.AFFINE, (a,b,c,d,e,f), resample=Image.BILINEAR)

def get_rect_from_bbox(bbox):
    if len(bbox) == 4:
        left = min(bbox[0][0], bbox[3][0])
        top = min(bbox[0][1], bbox[1][1])
        right = max(bbox[1][0], bbox[2][0])
        bottom = max(bbox[2][1], bbox[3][1])
    else:
        left = min(bbox[0], bbox[6])
        top = min(bbox[1], bbox[3])
        right = max(bbox[2], bbox[4])
        bottom = max(bbox[5], bbox[7])

    return (left, top, right, bottom)

def find_coeffs(pb, pa):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def perspective_transform(src_img, bbox):
    """
    To apply a perspective transformation you first have to know four points in a plane A that will be mapped to
    four points in a plane B. With those points, you can derive the homographic transform. By doing this, you obtain
    your 8 coefficients and the transformation can take place. 
    https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil/14178717#14178717
    """
    if len(bbox) == 4:  # if bbox is expressed as: x,y,w,h, convert it to four points
        x,y,w,h = bbox
        tmp_bbox = []
        tmp_bbox.append((x,y))
        tmp_bbox.append((x+w,y))
        tmp_bbox.append((x+w,y+h))
        tmp_bbox.append((x, y+h))
        bbox = tmp_bbox
    new_bbox = []
    for i in range(0, len(bbox)):
        new_x = (bbox[i][0] + randint(-4,4))
        new_y = (bbox[i][1] + randint(-4,4))
        new_bbox.append((new_x, new_y))

    max_width, max_height = src_img.size

    coeffs = find_coeffs(bbox, new_bbox)
    warped_img = src_img.transform((max_width, max_height), Image.PERSPECTIVE, coeffs, Image.NEAREST)

    return warped_img, new_bbox

def print_text(src_img, text_pos, text, color, font):
    painter = ImageDraw.Draw(src_img)
    painter.text(text_pos, text, fill=color, font=font)

    return src_img

def get_vertical_text_size(font, text):
    text_width = 0
    text_height = 0

    for char in text:
        char_width, char_height = font.getsize(char)
        if char_width > text_width:
            text_width = char_width
        text_height += char_height

    return text_width, text_height

def paste_vertical_text(template_img, text, color, font_file, font_size):
    text = text.decode("utf-8")
    if len(text) > 10:
        print "Warning: Print vertical text's len is bigger than 10!"
        return None, None
    try:
        width, height = template_img.size
        dst_img = template_img.copy()
    except Exception as e:
        print str(e)
        return None, None
    painter = ImageDraw.Draw(dst_img)
    font = ImageFont.truetype(font_file, font_size)

    text_width, text_height = get_vertical_text_size(font, text)
    if text_width + 10 > width or text_height + 10 > height:
        #print("The text is too long:%s, font_size=%d, width=%d, height=%d, text_width=%d, text_height=%d" %(text, font_size, width, height, text_width, text_height))
        return None, None
    if text_width == 0 or text_height == 0:
        #print("Text width or height is 0, text_width = %d, text_height = %d, %s" %(text_width, text_height, text))
        return None, None

    x = randint(0, width - text_width - 10)
    y = randint(0, height - text_height - 10)
    left = x
    top = y

    for char in text:
        dst_img = print_text(dst_img, (x,y), char, color, font)
        char_width, char_height = font.getsize(char)
        y += char_height

    return dst_img, (left,top,text_width, text_height)

def paste_text(template_img, text, color, font_file, font_size):
    text = text.decode("utf-8")
    #if len(text) <= 3:
    #    return None, None
    try:
        width, height = template_img.size
        dst_img = template_img.copy()
    except Exception as e:
        print str(e)
        return None, None
    font = ImageFont.truetype(font_file, font_size)

    text_width, text_height = font.getsize(text)
    if text_width + 10 > width or text_height + 10 > height:
        #print("The text is too long:%s, font_size=%d, width=%d, height=%d, text_width=%d, text_height=%d" %(text, font_size, width, height, text_width, text_height))
        return None, None
    if text_width == 0 or text_height == 0:
        #print("Text width or height is 0, text_width = %d, text_height = %d, %s" %(text_width, text_height, text))
        return None, None

    x = randint(0, width - text_width - 10)
    y = randint(0, height - text_height - 10)

    # 增加字体特效，如：阴影、描边
    effect = randint(0,15)
    all_shift = []
    if effect == 0: # 描边
        shift_pixel = randint(1,2)
        all_shift.append((-shift_pixel, 0))
        all_shift.append((shift_pixel,0))
        all_shift.append((0, -shift_pixel))
        all_shift.append((0, shift_pixel))
        all_shift.append((-shift_pixel, -shift_pixel))
        all_shift.append((shift_pixel, shift_pixel))
        all_shift.append((shift_pixel, -shift_pixel))
        all_shift.append((-shift_pixel, shift_pixel))
    elif effect == 1:   # 阴影
        shift_pixel = randint(1,1)
        all_shift.append((shift_pixel,shift_pixel))

    # Should the effect_color be different with font color?
    color_shift = 255
    r = randint(0, color_shift)
    g = randint(0, color_shift)
    b = randint(0, color_shift)
    effect_color = (r,g,b)
    for shift in all_shift:
        x1 = x+shift[0]
        y1 = y+shift[1]
        dst_img =  print_text(dst_img, (x1,y1), text, effect_color, font)

    dst_img = print_text(dst_img, (x,y), text, color, font)

    return dst_img, (x,y,text_width, text_height)

def crop_img_with_border_disturb(src_img, text_rect, horizontal=True):
    img_w, img_h = src_img.size

    left, top, right, bottom = get_rect_from_bbox(text_rect)

    if horizontal == True:
        min_disturb = 0
        max_disturb = 20
        h_max_disturb = 6
    else:
        min_disturb = 0
        max_disturb = 12
        h_max_disturb = 20

    left_border = randint(min_disturb, max_disturb)
    right_border = randint(min_disturb, max_disturb)
    top_border = randint(min_disturb, h_max_disturb)
    bottom_border = randint(min_disturb, h_max_disturb)

    left   = left - left_border
    right  = right + right_border
    top    = top - top_border
    bottom = bottom + bottom_border

    if left < 0: left = 0
    if right > img_w - 1: right = img_w -1
    if top < 0: top = 0
    if bottom > img_h -1: bottom = img_h - 1

    crop_img = src_img.crop((left, top, right, bottom))

    return crop_img

def salt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声

    return img_
def add_salt_pepper(img):
    SNR = random.uniform(0.95,1.0)
    img = np.array(img)
    img_s = salt_pepper(img.transpose(2, 1, 0), SNR)
    img_s = img_s.transpose(2, 1, 0)
    img_s=img_s[:,:,::-1]
    img_s = Image.fromarray(img_s, mode='RGB')

    return img_s

def Gauss_noise(img):
    img = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sigma = randint(4,36)  #可以调节方差范围
            g=random.gauss(0,sigma)
            r1=np.where((g+img[i,j])>255,255,(g+img[i,j]))
            r2=np.where(r1<0,0,r1)
            img[i,j]=np.round(r2)
    img_s = Image.fromarray(img, mode='RGB')
    return img_s

def gauss_noise(image):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image)
    image = np.array(image/255, dtype=float)
    var = random.uniform(0.0004,0.001)
    noise = np.random.normal(0, var ** 0.5, image.shape)
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    out = Image.fromarray(out, mode='RGB')
    #cv2.imshow("gauss", out)
    return out

def add_effects_to_image(src_img, bbox, horizontal=True):
    """
    1. Apply perspective transform or affine transform;
    2. Blur image;
    3. Crop text image from source image, using border disturbing;
    """
    # perspective transform
    #src_img, bbox = perspective_transform(src_img, bbox)
    
    bbox = convert_rect_style(bbox)
    # affine transform, set the probability to 2%
    if randint(0, 49) == 0:
        # sx, sy: resize scale, [0.8, 1.2]
        # angle: rotate angle, [-5, 5]
        sx = randint(8, 12)/10.
        sy = randint(8, 12)/10.
        angle = randint(-50, 50)/10.
        bbox, src_img = affine_transform(src_img, sx, sy, angle, bbox)

    # 加入椒盐噪声和高斯噪声
    #if randint(0,20) == 0:
    #    src_img = add_salt_pepper(src_img)
    if randint(0,10) == 0:
        src_img = gauss_noise(src_img)
    # Attention: It is not allowed to set big blur radius!
    blur_radius = randint(0,1)
    blur_img = src_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    crop_img = crop_img_with_border_disturb(blur_img, bbox, horizontal)

    return crop_img, blur_img
