import numpy as np

class PostProcessing(object):
	def __init__(self, score_thresh):
		"""
		score_thresh: label prob higher than the thresh is valid for post process
		:param score_thresh: 
		"""
		self.score_thresh = score_thresh

	def __call__(self, video_timestamps, video_classification_result):
		video_labels = []
		label_duration = []
		label_prob = []

		old_label = None; old_duration = []; old_probs = []
		for idx, batch_classification_result in enumerate(video_classification_result):
			for label, prob in batch_classification_result.items():
				if prob < self.score_thresh:
					continue
				if label == old_label:
					old_duration[1]=video_timestamps[idx][-1]
					old_probs.append(prob)
				else:
					if old_label is not None:
						video_labels.append(old_label)
						label_duration.append(old_duration)
						label_prob.append(np.mean(old_probs))
						old_probs = []
					old_label = label
					old_duration = [video_timestamps[idx][0], video_timestamps[idx][-1]]
					old_probs.append(prob)

		if old_label is not None:
			video_labels.append(old_label)
			label_duration.append(old_duration)
			label_prob.append(np.mean(old_probs))

		return video_labels, label_duration, label_prob
