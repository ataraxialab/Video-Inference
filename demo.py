import argparse
from video import Video
from featureExtract import FeatureExtraction
from featureCoding import FeatureCoding
from postProcessing import PostProcessing
from videoInfer import VideoInfer
from config import config


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

def init(args):
	feature_extract = FeatureExtraction(modelPrototxt=config.FEATURE_EXTRACTION.MODEL_PROTOTXT,
	                                    modelFile=config.FEATURE_EXTRACTION.MODEL_FILE,
	                                    featureLayer=config.FEATURE_EXTRACTION.FEATURE_LAYER, gpu_id=args.gpu_id)
	feature_coding = FeatureCoding(featureDim=config.FEATURE_CODING.FEATURE_DIM, batchsize=args.frame_group, modelPrefix=config.FEATURE_CODING.MODEL_PREFIX,
	                               modelEpoch=config.FEATURE_CODING.MODEL_EPOCH, synset=config.FEATURE_CODING.SYNSET, gpu_id=args.gpu_id)
	post_processing = PostProcessing(score_thresh=args.display_score_thresh)
	video_infer = VideoInfer(feature_extract, feature_coding, post_processing)

	return video_infer


if __name__ == '__main__':

	args = parse_args()
	print('Called with argument:', args)
	video_infer_handler = init(args)
	video = Video(args.video_path, step=args.step, frame_group_len=args.frame_group)

	video_labels, label_duration, label_prob = video_infer_handler.infer(video)
	video_infer_handler.composite_video(video, args.composite_video_name)
