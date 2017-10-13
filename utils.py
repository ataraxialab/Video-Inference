from collections import deque
import numpy as np
import cv2
import subprocess
import shutil

def center_crop_images(images, crop_dims):
    """
    Crop images into center.
    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (height x width x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    crops_ix = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = crops_ix[0]

    # Extract crops
    crop_images = deque([], len(images))

    for im in images:
        crop_image = im[int(crops_ix[0]):int(crops_ix[2]), int(crops_ix[1]):int(crops_ix[3]), :]
        crop_images.append(crop_image)

    return crop_images


class Composite_Video(object):
    def __init__(self, videoname , fps , framesize, ffmpeg='ffmpeg'):
        self.ffmpeg = ffmpeg
        self._img_prefix = "tmp/comp_img%d.jpg"
        self._img_idx = 0
        self._video_name = videoname
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._fontScale = 2
        self._fontThickness = 1
        self._fontColor = (255, 255, 255)
        textsize = cv2.getTextSize("test", self.font, self.fontScale, self.fontThickness)[0]
        self._text_height = textsize[1]
        self._text_line_gap = 5
        self._fps = fps
        self._width, self._height = framesize
        self._text_bottom_left_corner = (10, self.height - 10)

    def _draw_text(self, image, text_list):
        """Draws texts to a given image.
        Returns copy of image, original is not modified.
        """
        im = image.copy()
        current_x, current_y= self._text_bottom_left_corner
        for i in range(len(text_list),-1,-1):
            current_y = current_y - self._text_height
            text_top_left = (current_x, current_y)
            cv2.putText(im, text_list[i],
                        text_top_left,
                        self._font,
                        self._fontScale,
                        self._fontColor,
                        self._fontThickness)
            current_y = current_y - self._text_line_gap

        return im

    def _add_frame(self, frames, text_list):
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_with_text = self._draw_text(bgr, text_list)
            cv2.imwrite(self._img_prefix.format(self.img_idx), img_with_text)
            self._img_idx += 1

    def _composite_video(self, del_frames = True):
        cmd = [self.ffmpeg, "-framerate", self._fps,
               "-i", self._img_prefix,
               self._video_name]
        subprocess.call(cmd)

        if del_frames:
            shutil.rmtree('tmp')


