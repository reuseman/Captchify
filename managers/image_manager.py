import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_isodata, threshold_minimum, sobel
from skimage.measure import label, regionprops
from skimage.morphology import watershed, square, disk, erosion as erosion_filter
from skimage.filters.rank import median


class ImageManager:
    # TODO allargare in resize
    debug_fixed_bigger_than_4 = 0

    @staticmethod
    def get_binarized_image(image):
        threshold = 0
        try:
            threshold = threshold_minimum(image)
            threshold = np.add(threshold, 0.10009)
        except RuntimeError:
            print("-    [WARNING] minimum algorithm cannot be used, isodata will be used instead")
            threshold = threshold_isodata(image)
        finally:
            return image > threshold

    @staticmethod
    def get_erosioned_image(image, value=1):
        return erosion_filter(image, selem=square(value))

    @staticmethod
    def get_labeled_matrix(image):
        # Matrix of zeros with the same shape as the image, set value 1 if the original pixel is white, otherwise 2
        markers = np.zeros_like(image)
        markers[image > 0.1] = 1
        markers[image <= 0.1] = 2

        # Sobel use some sort of gradient and creates an elevation map that will be used to fill the different segments
        elevation_map = sobel(image)
        segmentation = watershed(elevation_map, markers)
        segmentation = ndi.binary_fill_holes(segmentation - 1)

        # Labeled is a matrix of the original image with 0 for the white, 1 to N for the different segments
        return label(segmentation)

    @staticmethod
    def get_segments(image, original=None):
        cropped_images = []
        labeled = ImageManager.get_labeled_matrix(image)
        for region_index, region in enumerate(regionprops(labeled)):
            if region.area < 15:
                continue

            min_row, min_col, max_row, max_col = region.bbox
            width = max_col - min_col
            height = max_row - min_row

            if original is not None:
                temp = np.copy(original[min_row:max_row, min_col:max_col])
            else:
                temp = np.copy(image[min_row:max_row, min_col:max_col])

            sub_labeled = labeled[min_row:max_row, min_col:max_col]

            # Every pixel that does not belong to the current region will be sett as white (other letters that enter in
            # the current region)
            temp[sub_labeled != region_index + 1] = 1

            # temp = resize(temp, (40, 40), anti_aliasing=True)
            # The tuple contains the image and the min_col, the reason why for the latter is that is needed to sort
            # and don't lost the mapping in the dataset between captcha and the relative solution
            cropped_images.append((temp, min_col))

        # Sort by colmin and then remove the tuple (image, colmin) and obtain only image as element of the list
        cropped_images.sort(key=lambda cropped: cropped[1])
        return list(map(lambda value: value[0], cropped_images))

    @staticmethod
    def split_segments_with_two_letters(segments):
        # Obtain the index of the segment that contain two letters
        max_width_index, max_width = -1, -1
        for i, seg in enumerate(segments):
            if seg.shape[1] > max_width:
                max_width = seg.shape[1]
                max_width_index = i

        # Obtain the index of the column where the cut should be done.
        # Sum every column, and pick the one with the maximum value (more white pixels). A weight will be applied
        segment_to_cut = segments[max_width_index]
        sum_col = np.sum(segment_to_cut, axis=0)

        # Distance between the middle
        width = segment_to_cut.shape[1]
        mid_point = width // 2
        distances_from_mid_point = [abs(mid_point - 1 - n) for n in range(0, width)]
        distances_from_mid_point = np.multiply(distances_from_mid_point, 0.7)
        sum_col = np.subtract(sum_col, distances_from_mid_point)

        # Obtain the index of the column that contains more white pixels
        max_sum_value, max_sum_index = -1, -1
        for i, value in enumerate(sum_col):
            if value > max_sum_value:
                max_sum_value = value
                max_sum_index = i

        new_segments = []
        for i in range(0, len(segments)):
            if i != max_width_index:
                new_segments.append(segments[i])
            else:
                image1 = segments[i][:, :max_sum_index]
                image2 = segments[i][:, max_sum_index:]

                new_segments.append(median(image1, disk(1)))
                new_segments.append(median(image2, disk(1)))

        return new_segments

    @staticmethod
    def get_n_segments(image, number, debug_image_name="image"):
        fixed_length_of_segments = False
        segments = ImageManager.get_segments(image)

        if len(segments) == 3:
            segments = ImageManager.split_segments_with_two_letters(segments)
            result = "done" if len(segments) == number else "fail"
            print("     [SPLITTING] > {}, now contains {} segments".format(result, len(segments)))
            fixed_length_of_segments = True if result == "done" else False

        success = True if len(segments) == number else False
        return success, segments, fixed_length_of_segments
