import time
import argparse
from video import Video
from featureExtract import FeatureExtraction
from featureCoding import FeatureCoding
from utils import Composite_Video


def parse_args():
	parser = argparse.ArgumentParser(description='Video Inference Demo')
	parser.add_argument('--video_path', help='video to be classified', default='test.avi', type=str)
	parser.add_argument('--step', help='Iterate frames every `step` seconds. Defaults to iterating every frame.', default=None, type=float)
	parser.add_argument('--frame_group', help='number of frames to be grouped as one classification input', default=1, type=int)
	parser.add_argument('--gpu_id', help='which gpu to use', default=0, type=int)
	parser.add_argument('--composite_video', help='composite a new video with video inference result.', action='store_true')
	parser.add_argument('--composite_video_name', help='new video name', default='newvideo.mp4', type=str)
	parser.add_argument('--display_score_thresh', help='label prob higher than the thresh can be displayed', default=0.1, type=float)


	args = parser.parse_args()
	return args


if __name__ == '__main__':

	args = parse_args()
	print('Called with argument:', args)
	video = Video(args.video_path, step=args.step, frame_group_len=args.frame_group)
	feature_extract = FeatureExtraction(video, modelPrototxt='./models/SENet.prototxt', modelFile='./models/SENet.caffemodel',
	                             featureLayer='pool5/7x7_s1', gpu_id=0)
	featurecoding = FeatureCoding(featureDim=512, batchsize=args.frame_group, modelPrefix='models/netvlad', modelEpoch=50, synset='lsvc_class_index.txt', gpu_id=0)

	if args.composite_video:
		newvideo = Composite_Video(videoname=args.composite_video_name, fps = 1. /video.step, framesize = video._size)

	t1 = time.time()
	for timestamps, frames, classification_result in featurecoding(feature_extract, topN=5):
		t2 = time.time()
		print "time cost: %f, results in timeduration:(%f~%f)s\n"%(t2-t1,timestamps[0],timestamps[-1])
		texts = []
		for label,prob in classification_result.items():
			text = "{0}:{1:0.2f}%".format(label,prob*100)
			if prob > args.display_score_thresh:
				texts.append(text)
			print text
		print "--"*10
		t1 = time.time()

		if args.composite_video:
			newvideo._add_frame(frames, texts)

	if args.composite_video:
		print "generating new video ..."
		newvideo._composite_video()
		print args.composite_video_name + " is generated."

