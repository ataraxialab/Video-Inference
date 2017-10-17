import time
from utils import Composite_Video

class VideoInfer(object):
	def __init__(self, FeatureExtraction_obj, FeatureCoding_obj, PostProcessing_obj):
		self.feature_extraction = FeatureExtraction_obj
		self.feature_coding = FeatureCoding_obj
		self.post_processing = PostProcessing_obj

	def infer(self, video):
		video_timestamps = []
		video_classification_result = []
		t1 = time.time()
		for batch_timestamps, _, batch_classification_result in self.feature_coding(self.feature_extraction(video), topN=1):
			t2 = time.time()
			print "time cost: %f, results in timeduration:(%f~%f)s\n" % (t2 - t1, batch_timestamps[0], batch_timestamps[-1])
			for label, prob in batch_classification_result.items():
				print "{0}:{1:0.2f}%".format(label, prob * 100)
			print "--" * 10
			video_timestamps.append(batch_timestamps)
			video_classification_result.append(batch_classification_result)
			t1 = time.time()

		label_durations, video_labels, label_probs = self.post_processing(video_timestamps, video_classification_result)
		return label_durations, video_labels, label_probs

	def composite_video(self, video, composite_video_name, display_score_thresh):
		newvideo = Composite_Video(videoname=composite_video_name, fps=1. / video.step, framesize=video._size)
		t1 = time.time()
		for batch_timestamps, batch_frames, batch_classification_result in self.feature_coding(self.feature_extraction(video), topN=5):
			t2 = time.time()
			print "time cost: %f, results in timeduration:(%f~%f)s\n" % (t2 - t1, batch_timestamps[0], batch_timestamps[-1])
			texts = []
			for label, prob in batch_classification_result.items():
				text = "{0}:{1:0.2f}%".format(label, prob * 100)
				if prob > display_score_thresh:
					texts.append(text)
				print text
			print "--" * 10
			t1 = time.time()
			newvideo._add_frame(batch_frames, texts)

		print "generating new video ..."
		newvideo._composite_video()
		print composite_video_name + " is generated."
