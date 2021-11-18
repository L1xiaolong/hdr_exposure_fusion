import cv2
import numpy as np
import time


def align(images):
    if not isinstance(images, list) or len(images) < 2:
        print("-- image align: input has to be a list of at least two images")
        return None
    t1 = time.time()
    # select base image
    hist_std = []
    for i in range(len(images)):
        hist_std.append(get_hist_std(images[i]))

    t2 = time.time()
    print("-- calculate images histogram in " + str(round(t2 - t1, 3)) + "s")

    base_index = hist_std.index(min(hist_std))

    print("-- the base image for aligning is images[" + str(base_index) + "]")

    images[0], images[base_index] = images[base_index], images[0]

    gray_images = []
    for image in images:
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    model_image = gray_images[0]

    sz = model_image.shape

    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    aligned_images = [images[0]]
    t = time.time()
    img_size = len(images)
    for i in range(1, img_size):
        (cc, warp_matrix) = cv2.findTransformECC(model_image, gray_images[i], warp_matrix, warp_mode, criteria,
                                                 inputMask=None, gaussFiltSize=5)
        duration = round(time.time() - t, 3)
        t = time.time()
        print("-- [" + str(i) + "/" + str(img_size) + "] " + "find transform ECC in " + str(duration) + "s")

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            aligned_image = cv2.warpPerspective(images[i], warp_matrix, (sz[1], sz[0]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            aligned_image = cv2.warpAffine(images[i], warp_matrix, (sz[1], sz[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        duration = round(time.time() - t, 3)
        t = time.time()
        print("-- [" + str(i) + "/" + str(img_size) + "] " + "warp in " + str(duration) + "s")

        aligned_images.append(aligned_image)

    aligned_images[0], aligned_images[base_index] = aligned_images[base_index], aligned_images[0]

    return aligned_images


def get_hist_std(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return np.std(hist)
