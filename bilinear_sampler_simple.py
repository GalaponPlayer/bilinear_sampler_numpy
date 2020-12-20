import fire
import numpy as np
import argparse
from PIL import Image

def bilinear_sampler(img, depth, focal=707.0, base=0.54, ifBTS=True):
    if ifBTS:
        depth = np.array(depth) / 256.0
    else:
        depth = np.array(depth)
    
    input_h, input_w = depth.shape

    disp = base * focal / depth
    x_offset = disp.astype(np.int)

    left = np.array(img.resize((input_w, input_h)))
    left = left.transpose(2,0,1)

    w_idx = np.arange(0, input_w)
    idx = np.repeat(w_idx[None, :], input_h, axis=0)

    x0 = idx + x_offset
    x0 = np.clip(x0, 0, input_w - 1)

    x = idx + disp
    x = np.clip(x, 0.0, input_w - 1.0)

    x1 = x0 + 1
    x1 = np.clip(x1, 0, input_w - 1)

    pix_l = np.zeros((3, input_h, input_w))
    pix_r = np.zeros((3, input_h, input_w))

    for i in range(3):
        for j in range(input_h):
            for k in range(input_w):
                pix_l[i][j][k] = left[i][j][x0[j][k]]
    for i in range(3):
        for j in range(input_h):
            for k in range(input_w):
                pix_r[i][j][k] = left[i][j][x1[j][k]]
    
    x0 = np.clip(x0, 0, input_w - 2)
    dist_l = x - x0
    dist_r = x1 - x

    output = dist_r * pix_l + dist_l * pix_r

    output = output.transpose(1,2,0)
    right_img = Image.fromarray(np.uint8(output))

    return right_img

if __name__ == "__main__":
    img = Image.open('./test/0000000096.png')
    depth = Image.open('./test/2011_09_26_drive_0036_sync_0000000096.png')

    right = bilinear_sampler(img,depth)

    right.save('./test/right.png')