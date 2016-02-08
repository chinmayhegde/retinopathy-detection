
import cv2
# import numpy
import csv


if __name__ == '__main__':
    # Get image names and associate
    names = {i: [] for i in range(5)}
    with open('subsetcsv.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            names[int(row[1])].append('train/' + row[0] + '.jpeg')

    # TODO: we should experiment with these
    win_size = (16, 16)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
                            num_bins)

    images = {i: [] for i in range(5)}
    for classification, image_names in names.iteritems():
        for image_name in image_names:
            # median image size in (2592, 3888)
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (2592, 3888))
            image = hog.compute(image)
            print image.shape
            images[classification].append(image)

    for classiction in images.iterkeys():
        print len(images[classification])


"""
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
"""

