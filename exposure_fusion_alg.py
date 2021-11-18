import cv2
import numpy as np
import time


def exposure_fusion(images, depth=7):
    weights = weights_cal(images)
    res = multi_scale_fusion(images, weights, depth)

    return weights, res


def weights_cal(images):
    w_c, w_s, w_e = 1, 1, 1

    H = np.shape(images[0])[0]
    W = np.shape(images[0])[1]
    D = len(images)
    weight = np.zeros((H, W, D), dtype='float64')

    t = time.time()
    img_size = len(images)
    count = 1
    i = 0
    for image in images:
        # W_C
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image.astype('float64'), cv2.CV_64F)
        weight_c = np.absolute(laplacian)
        weight_c = cv2.medianBlur(weight_c.astype('float32'), 5)
        weight_c = weight_c.astype('float64')
        #cv2.imwrite("c.png", weight_c*255)

        # W_S
        weight_s = np.std(image, axis=2, dtype=np.float32).astype('float64')
        #cv2.imwrite("s.png", weight_s * 255)

        # W_E
        sigma = 0.2
        image = image.astype('float64') / 255
        gauss_curve = lambda i: np.exp(
            -((i - 0.5) ** 2) / (2 * sigma * sigma))
        R_gauss_curve = gauss_curve(image[:, :, 2])
        G_gauss_curve = gauss_curve(image[:, :, 1])
        B_gauss_curve = gauss_curve(image[:, :, 0])
        weight_e = (R_gauss_curve * G_gauss_curve * B_gauss_curve).astype('float64')
        #cv2.imwrite("e.png", weight_e * 255)

        epsilon = 1e-10
        weight[:, :, i] = (np.power(weight_c, w_c) * np.power(weight_s, w_s) * np.power(weight_e, w_e)) + epsilon
        i += 1

        duration = round(time.time() - t, 3)
        t = time.time()
        print("-- [" + str(count) + "/" + str(img_size) + "] " + "weight map calculate in " + str(duration) + "s")
        count += 1

    # normalizations
    weight_sum = np.sum(weight, 2)
    for i in range(D):
        weight[:, :, i] = np.divide(weight[:, :, i], weight_sum)

    return weight


def multi_scale_fusion(images, weights, depth):
    t1 = time.time()
    laplacian_list = []
    for image in images:
        lap = laplacian_pyramid_gen(image.astype('float64'), depth)
        laplacian_list.append(lap)
    t2 = time.time()
    print("-- laplacian pyramid generate in " + str(round(t2 - t1, 3)) + "s")

    gaussian_list = []
    for i in range(len(images)):
        gau = gaussian_pyramid_gen(weights[:, :, i], depth)
        gaussian_list.append(gau)

    t3 = time.time()
    print("-- gaussian pyramid generate in " + str(round(t3 - t2, 3)) + "s")

    blendedPyramids = []
    for i in range(len(images)):
        blended_multires = []
        for j in range(depth + 1):
            blended_multires.append(laplacian_list[i][j] *
                                    np.dstack([gaussian_list[i][j], gaussian_list[i][j], gaussian_list[i][j]]))
        blendedPyramids.append(blended_multires)

    finalPyramid = []
    for i in range(depth + 1):
        intermediate = []
        tmp = np.zeros_like(blendedPyramids[0][i])
        for j in range(len(images)):
            tmp += np.array(blendedPyramids[j][i])
        intermediate.append(tmp)
        finalPyramid.append(intermediate)

    blended_final = np.array(finalPyramid[0][0])
    for i in range(depth):
        imgH = np.shape(images[0])[0]
        imgW = np.shape(images[0])[1]
        layerx = cv2.pyrUp(finalPyramid[i + 1][0])
        blended_final += cv2.resize(layerx, (imgW, imgH))

    blended_final[blended_final < 0] = 0
    blended_final[blended_final > 255] = 255

    t3 = time.time()
    print("-- image fusion in " + str(round(t3 - t2, 3)) + "s")

    return blended_final.astype('uint8')


def gaussian_pyramid_gen(image, depth):
    G = image.copy()
    gaussian = [G]
    for i in range(depth):
        G = cv2.pyrDown(G)
        gaussian.append(G)
    return gaussian


def laplacian_pyramid_gen(image, depth):
    gaussian = gaussian_pyramid_gen(image, depth)
    laplacian = [gaussian[-1]]

    for i in range(depth, 0, -1):
        H = np.shape(gaussian[i - 1])[0]
        W = np.shape(gaussian[i - 1])[1]
        temp_upsampling = cv2.resize(cv2.pyrUp(gaussian[i]), (W, H))
        tmp_lap = cv2.subtract(gaussian[i - 1], temp_upsampling)
        laplacian.append(tmp_lap)

    laplacian.reverse()
    return laplacian




