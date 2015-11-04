import cv2
import numpy


def get_default_image_data():
    img = cv2.imread('digits.png', cv2.IMREAD_GRAYSCALE)

    # We know the sub-images are 20x20, and there are 50x100 sub-images in the
    # larger image. Works for this one, might need to make better in future.
    cells_list = [numpy.hsplit(row, 100) for row in numpy.vsplit(img, 50)]
    cells = numpy.array(cells_list)  # 50x100 array of 20x20 arrays

    # Divide dataset in half, half to train and half to test
    # Make each half a 2D array, the images flattened to a single array
    # and each entry in the first array is an image.
    train_data = cells[:, :50].reshape((-1, 400)).astype(numpy.float32)
    test_data = cells[:, 50:100].reshape((-1, 400)).astype(numpy.float32)

    # Form expected, it will be the same for each piece as we divided in half
    expected = numpy.repeat(numpy.arange(10), 250)[:, numpy.newaxis]

    return train_data, test_data, expected


def get_default_image_transformed(bin_n=16, size=20):
    img = cv2.imread('digits.png', 0)
    cells = [numpy.hsplit(row, 100) for row in numpy.vsplit(img, 50)]

    transformed = transform(cells)
    train_data = numpy.float32([t[50:] for t in transformed])
    train_data = train_data.reshape(-1, bin_n * 4)
    test_data = numpy.float32([t[:50] for t in transformed])
    test_data = test_data.reshape(-1, bin_n * 4)
    expected = numpy.repeat(numpy.arange(10), 250)[:, numpy.newaxis]
    return train_data, test_data, expected


def deskew(img, size=20):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = numpy.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    img = cv2.warpAffine(img, M, (size, size), flags=affine_flags)
    return img


def hog(img, bin_n=16):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = numpy.int32(bin_n*ang/(2 * numpy.pi))
    # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], \
        bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [numpy.bincount(b.ravel(), m.ravel(), bin_n)
             for b, m in zip(bin_cells, mag_cells)]
    hist = numpy.hstack(hists)     # hist is a 64 bit vector
    return hist


def transform(img):
    return [[hog(deskew(cell)) for cell in row] for row in img]
