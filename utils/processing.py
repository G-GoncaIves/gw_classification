from random import randint
from typing import Tuple

import numpy as np
import numpy.typing as npt


class Inspection():
    """
    This class is made to store methods than find properties of a array.
    """

    def get_window_idx(array: npt.ArrayLike, threshold: float) -> Tuple[int]:
        """
        Finds the index of the first and last values of the array above
        the specified threshold. If no idxs are found, returns None.

        Input:

            array: numpy.array() - array with non-zero values.

            threshold: float - value above which, amplitude becomes relevant.

        Output:

            left_idx: int - index of first element
                        whose value is greater than the threshold.

            right_idx: int - index of last element
                        whose value is greater than the threshold.

            or,

            None - if no valid idx are found.

        """
        array = np.asarray(array)
        threshold = float(threshold)
        assert len(array.shape) == 1, f"Unexpected input shape.\n\tGot: {len(array.shape)}\n\tExpected: (n,)"

        idxs = np.where(array > threshold)
        if len(idxs) > 1:
            return idxs[0], idxs[-1]
        else:
            return 0, len(array) - 1


class ReShape():
    """
    This class is created to store methods that manipulate arrays.
    """

    @staticmethod
    def reshape_array(array: npt.ArrayLike, out_size: int, crop_radius: int, centered: bool) -> npt.ArrayLike:
        """
        Reshapes the array with use of cropping and padding.

        Input:

            array: numpy.array - array with non-zero values.

            out_size: int - Desired output size.

            crop_radius: int - Radius used to set the crop(or crop) window.

            centered: bool - Whether the number of zeros added to the right and left are equal.

        Output:

            numpy.array - array with length out_size

        """
        array = np.asarray(array)
        assert out_size > 2 * crop_radius, f"crop window larger than desired output size. \n\tDesired Output Size: {out_size} \n\tcrop Window Size: {2 * crop_radius}"

        cropd_array = ReShape.crop_array(array, crop_radius)
        padded_array = ReShape.pad_array(cropd_array, out_size, centered)

        return padded_array

    @staticmethod
    def crop_array(array: npt.ArrayLike, crop_radius: int) -> npt.ArrayLike:
        """
        Crops the array around its peak with a given radius.

        Input:

            array: numpy.array - array with non-zero values.

            crop_radius: int - Radius used to set the crop(or crop) window.

        Output:

            numpy.array - array with length 2*crop_radius
        """
        assert len(
            array) > 2 * crop_radius, f"Crop window larger than array size. \n\tArray Size: {len(array)} \n\tCrop Window Size: {2 * crop_radius}"

        peak_idx = np.argmax(array)
        peak_idx = int(peak_idx)
        left_zero_idx, right_zero_idx = Inspection.get_window_idx(array, threshold=1e-4)
        left_zero_idx, right_zero_idx = int(left_zero_idx), int(right_zero_idx)

        if left_zero_idx > peak_idx - crop_radius:
            left_crop_idx = left_zero_idx
            left_extra = left_zero_idx - peak_idx + crop_radius
        else:
            left_crop_idx = peak_idx - crop_radius
            left_extra = 0

        if right_zero_idx < peak_idx + crop_radius:
            right_crop_idx = right_zero_idx
            right_extra = peak_idx + crop_radius - right_zero_idx
        else:
            right_crop_idx = peak_idx + crop_radius
            right_extra = 0

        if (not left_extra == 0 and right_extra == 0) or (left_extra == 0 and not right_extra == 0):
            if left_crop_idx - right_extra >= 0:
                left_crop_idx -= right_extra

            if right_crop_idx + left_extra < len(array):
                right_crop_idx += left_extra

        return array[int(left_crop_idx) + 1:int(right_crop_idx) + 1]

    @staticmethod
    def pad_array(array: npt.ArrayLike, out_size: int, centered: bool) -> npt.ArrayLike:
        """
        Adds zeros to the input array, untill the desired output size is achieved.

        Input:

            array: numpy.array - array with non-zero values.

            out_size: int - Desired output size.

            centered: bool - Whether the number of zeros added to the right and left are equal.

        Output:

            numpy.array - array with length out_size.
        """
        assert len(
            array) < out_size, f"array size larger than desired output size. \n\tarray Size: {len(array)} \n\tDesired Output Size: {out_size}"

        left_pad = (out_size - len(array)) // 2
        right_pad = out_size - len(array) - left_pad

        if not centered:
            shift = randint(-left_pad + 2, left_pad - 2)
            left_pad += shift
            right_pad += -shift

        padded_array = np.pad(array, (left_pad, right_pad), 'constant', constant_values=(0, 0))
        return padded_array