'''
adapted from https://github.com/ouyangzhibo/Image_Foveation_Python
Perry, Jeffrey S., and Wilson S. Geisler. "Gaze-contingent real-time simulation of arbitrary visual fields." 
Human vision and electronic imaging VII. Vol. 4662. International Society for Optics and Photonics, 2002.

# pixels_per_degree = image height in pixels / visual angle in degrees
# you need to adjust k so that high resolution area equal to pixel size of visual angle 1-2 deg
e.g., python retina_transform_v2.py images/000000000139.jpg k=4, pixels_per_degree=11, resulting in 145 high-res pixels
e.g., python retina_transform_v2.py images/TP_bird_d300.png k=30 pixels_per_degree= 30, resulting in  1109 high-res pixels
'''
import cv2
import numpy as np
import sys


import cv2
import numpy as np
import sys

def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    
    
    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids

def foveat_img(im, fixs, k, pixels_per_degree, verbose=False):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    # get gaussian pyramids
    sigma=0.248 # constant by Perry and Geisler
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    
    # gaze-dependent resolution map (from 0 to 1)
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / pixels_per_degree
    alpha = 2.5 # The half-resolution constant of the human visual system, range of 2.0~2.5 deg.
    R = alpha / (theta + alpha)
    
    # transfer function
    Ts = []

    for i in range(1, prNum):
        Ts.append(np.exp(-0.5*(((2 ** (i)) * R / (sigma*pixels_per_degree)) ** 2*k )))

    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):       
        omega[i-1] =(sigma*pixels_per_degree)* np.sqrt(2*np.log(2)/k) /(2**(i))

    omega[omega>1] = 1
    omega[-1] = 0

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    if verbose:
        print('num of full-res pixel', np.sum(Ms[0] == 1))

    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Wrong format: python retina_transform.py [image_path] [k] [pixels_per_degree]")
        exit(-1)

    im_path = sys.argv[1]
    k = float(sys.argv[2])
    pixels_per_degree = float(sys.argv[3]) # image height in pixels / visual angle in degrees

    im = cv2.imread(im_path)
    # im = cv2.resize(im, (512, 320), cv2.INTER_CUBIC)
    xc, yc = int(im.shape[1]/2), int(im.shape[0]/2)

    im = foveat_img(im, [(xc, yc)], k, pixels_per_degree, verbose=True)

    cv2.imwrite(im_path.split('.')[0]+'_RT.jpg', im)
    print('img saved!')
