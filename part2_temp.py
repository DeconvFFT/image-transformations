import numpy as np
from re import S
from PIL import Image
from PIL import ImageFilter
import sys
import random
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from scipy import ndimage

def get_transform_mat(option, src, dest):
    if option==1:
        x1,y1 = src
        x1_, y1_ = dest
        H = np.array([[1,0,x1-x1_],[0,1,y1-y1_],[0,0,1]])
    elif option==2:
        x1,y1, x2, y2 = src
        x1_, y1_, x2_, y2_ = dest
        T = np.array([[1,0,x1-x1_],[0,1,y1-y1],[0,0,1]])
        T = T.reshape(3,3)

        m1 = (y2_-y1_)/(x2_-x1_)
        m2 = (y2-y1)/(x2-x1)

        rad = np.arctan(np.abs(m2-m1)/(1+m2*m1))
        angle = np.rad2deg(rad)
        sh1 = np.array([[1, -np.tan(angle/2),0],[0,1,0],[0,0,1]]).reshape(3,3)
        sh2 = np.array([[1, 0,0],[np.sin(angle),1,0],[0,0,1]]).reshape(3,3)
        sh3 = np.array([[1, -np.tan(angle/2),0],[0,1,0],[0,0,1]]).reshape(3,3)
        H = T  @ sh1 @sh2 @sh3
    elif option==3:
        x1, y1, x2, y2, x3, y3 = src
        x1_, y1_, x2_, y2_, x3_, y3_ = dest

        A = np.array([[x1,y1,1,0,0,0],[0,0,0,x1,y1,1],[x2,y2,1,0,0,0],[0,0,0,x2,y2,1],[x3,y3,1,0,0,0],[0,0,0,x3,y3,1]])
        b = np.array([x1_, y1_, x2_, y2_, x3_, y3_])
        H = np.linalg.solve(A,b)
        H = np.append(H,[0,0,1])
    else:
        x1,y1,x2,y2,x3,y3,x4,y4 = src
        x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_ = dest
        A = np.array([[x1,y1,1,0,0,0, -x1*x1_, -y1*x1_],
        [0,0,0,x1,y1,1, -x1*y1_, -y1*y1_],
        [x2,y2,1,0,0,0, -x2*x2_, -y2*x2_],
        [0,0,0,x2,y2,1, -x2*y2_, -y2*y2_],
        [x3,y3,1,0,0,0, -x3*x3_, -y3*x3_],
        [0,0,0,x3,y3,1, -x3*y3_, -y3*y3_], 
        [x4, y4, 1, 0,0, 0, -x4*x4_, -y4*x4_], 
        [0,0,0,x4, y4, 1, -x4*y4_, -y4*y4_]])
        b = np.array([x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_])
        H = np.linalg.solve(A,b)
        H = np.append(H, 1)

    H = H.reshape(3,3)
    return H


def bilinear_interpolation(image, x,y):

    # reference: https://en.wikipedia.org/wiki/Bilinear_interpolation#Example

    x1 = int(np.floor(x))
    x2 = x1 + 1
    y1 = int(np.floor(y))
    y2 = y1 + 1
    
    # get pixels at locations based on x1,y1,x2,y2

    img_a = image[y1,x1]
    img_b = image[y2,x1]
    img_c = image[y1,x2]
    img_d = image[y2,x2]

    # capture deltas for x and y
    dx = x - x1
    dy = y - y1

    point = img_a * (1-dx)*(1-dy) + img_b * (dy)* (1-dx) + img_c * (dx)*(1-dy) + img_d * dx*dy
    return np.round(point)

def inverse_warp(m,n,H, original_img):
    new_image = np.zeros((m,n, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)
    for r in range(m):
        for c in range(n):
            # adjust for offset
            coords = np.array([c,r,1])
            new_i, new_j, scale = H_inv @ coords
            new_i, new_j = new_i/scale, new_j/scale
            # perform bilinear interpolation
            if 0 <= new_j < (m - 1) and 0 <= new_i < (n - 1):
                new_image[r, c, :] =  bilinear_interpolation(original_img, new_i, new_j)
    return new_image

if __name__ == "__main__":
    # load an image
    option = int(sys.argv[1])
    image = Image.open(sys.argv[3])
    outpath = sys.argv[4]

    image_arr = np.array(image)
    src = []
    dest = []
    counter = 0
    points = sys.argv[5:]
    for i in range(0,len(points)-3,4):
        src.append(int(points[i]))
        src.append(int(points[i+1]))
        dest.append(int(points[i+2]))
        dest.append(int(points[i+3]))
    H = get_transform_mat(option,src, dest)

    H = np.linalg.inv(H)
    print(f'transformation matrix between source and destination images: \n{H}')

    inverse_image = inverse_warp(image_arr.shape[0], image_arr.shape[1],H,image_arr)
    inverse_image_pil = Image.fromarray((inverse_image).astype(np.uint8))

    inverse_image_pil.save(outpath)
    


    
