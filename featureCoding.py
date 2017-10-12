from featureExtract import FeatureExtraction
import mxnet as mx
import numpy as np
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


class FeatureCoding(object):
	"""
	extract features for video frames.

	Parameters
	-----------------
	featureDim: input feature dim for coding
	modelPrefix: models snapshot
	modelEpoch: models snapshot
	synset: class label file
	gpu_id: which gpu to use
	"""

	def __init__(self, featureDim=512, batchsize=1, modelPrefix='models/netvlad', modelEpoch=0, synset='lsvc_class_index.txt', gpu_id=0):
		self.batchsize = batchsize
		ctx = mx.gpu(gpu_id)
		sym, arg_params, aux_params = mx.model.load_checkpoint(modelPrefix, modelEpoch)
		mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
		mod.bind(for_training=False, data_shapes=[('data', (1, self.batchsize, featureDim))],
		         label_shapes=mod._label_shapes)
		mod.set_params(arg_params, aux_params, allow_missing=True)
		self.mod = mod
		self.featureDim = featureDim
		with open(synset, 'r') as f:
			labels = [l.strip().split('\t')[-1] for l in f]
		self.labels = labels

	def __call__(self, feature_extraction, topN = 5):
		for batch_timestamps, extracted_batch_feature in feature_extraction():
			feature = extracted_batch_feature.reshape(1,self.batchsize,-1)
			if feature.shape[-1] != self.featureDim:
				feature = self._cut_feature(feature)

			self.mod.forward(Batch([mx.nd.array(feature)]))
			prob = self.mod.get_outputs()[0].asnumpy()
			prob = np.squeeze(prob)
			sorted_idx = np.argsort(prob)[::-1]
			topN_result = dict()
			for i in sorted_idx[0:topN]:
				topN_result[self.labels[i]] = prob[i]

			yield batch_timestamps, topN_result


	def _cut_feature(self, source_fea):
		if source_fea.shape[-1] < self.featureDim:
			raise IOError(
				("FeatureCoding error: feature dimension (%d) is smaller than cut feature dimension (%d)"
				 % (source_fea.shape[-1], self.featureDim)))

		fea_len = int(source_fea.shape[-1] / self.featureDim)
		batchsize = source_fea.shape[1]
		randidx = np.random.randint(fea_len,size=batchsize)
		crop_fea = np.empty((1, self.batchsize, self.featureDim), dtype=np.float32)
		for ind, start_dim in enumerate(randidx):
			crop_fea[0,ind,:] = source_fea[0,ind,start_dim:start_dim+self.featureDim]

		return crop_fea


if __name__ == '__main__':
	from video import Video
	import time

	filename = "test.avi"
	video = Video(filename, frame_group_len=1)
	feature_extract = FeatureExtraction(video, modelPrototxt='./models/SENet.prototxt', modelFile='./models/SENet.caffemodel',
	                             featureLayer='pool5/7x7_s1', gpu_id=0)
	featurecoding = FeatureCoding(featureDim=512, batchsize=1, modelPrefix='models/netvlad', modelEpoch=50, synset='lsvc_class_index.txt', gpu_id=0)

	t1 = time.time()
	for timestamps, classification_result in featurecoding(feature_extract, topN=5):
		t2 = time.time()
		print "time cost: %f, results in timeduration:(%f~%f)s\n"%(t2-t1,timestamps[0],timestamps[-1])
		for label,prob in classification_result.items():
			print "%s:%f"%(label,prob)
		print "--"*10
		t1 = time.time()


