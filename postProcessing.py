import numpy as np
import time

class PostProcessing(object):
	def __init__(self, score_thresh):
		"""
		score_thresh: label prob higher than the thresh is valid for post process
		:param score_thresh: 
		"""
		self.score_thresh = score_thresh

	def __call__(self, feature_coding, feature_extract):
		video_labels = []
		label_duration = []
		label_prob = []

		old_label = None; old_duration = []; old_probs = []
		t1 = time.time()
		for batch_timestamps, _, batch_classification_result in feature_coding(feature_extract, topN=1):
			t2 = time.time()
			for label, prob in batch_classification_result.items():
				if prob < self.score_thresh:
					continue
				print "time cost: %f, top1 label in timeduration:(%.2f~%.2f)s is %s (prob:%.4f)" % (t2 - t1, batch_timestamps[0], batch_timestamps[-1], label, prob)
				if label == old_label:
					old_duration[1]=batch_timestamps[-1]
					old_probs.append(prob)
				else:
					if old_label is not None:
						video_labels.append(old_label)
						label_duration.append(old_duration)
						label_prob.append(np.mean(old_probs))
						old_probs = []
					old_label = label
					old_duration = [batch_timestamps[0], batch_timestamps[-1]]
					old_probs.append(prob)
			t1 = time.time()

		if old_label is not None:
			video_labels.append(old_label)
			label_duration.append(old_duration)
			label_prob.append(np.mean(old_probs))

		return video_labels, label_duration, label_prob


if __name__ == '__main__':
	from video import Video
	from featureExtract import FeatureExtraction
	from featureCoding import FeatureCoding

	filename = "test.avi"
	frame_group = 1
	video = Video(filename, step=1, frame_group_len=frame_group)
	feature_extract = FeatureExtraction(video, modelPrototxt='./models/SENet.prototxt', modelFile='./models/SENet.caffemodel',
	                             featureLayer='pool5/7x7_s1', gpu_id=0)
	feature_coding = FeatureCoding(featureDim=512, batchsize=frame_group, modelPrefix='models/netvlad', modelEpoch=50, synset='lsvc_class_index.txt', gpu_id=0)
	post_processing = PostProcessing(score_thresh=0.1)

	video_labels, label_duration, label_prob = post_processing(feature_coding, feature_extract)

	print "--"*10
	if len(video_labels) == 0 :
		print "this video is confusing, no label has a prob higher than score_thresh"
	else:
		print "video-level classification result:"
		for idx, video_label in enumerate(video_labels):
			print "label:{0}, duration:{1:0.2f}~{2:0.2f}, prob:{3:0.2f}%"\
				.format(video_label,label_duration[idx][0],label_duration[idx][-1],label_prob[idx]*100)
