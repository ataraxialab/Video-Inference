from featureExtract import FeatureExtraction
import mxnet as mx
import numpy as np
from collections import namedtuple, OrderedDict
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

	def __init__(self, featureDim, batchsize, modelPrefix, modelEpoch, synset, gpu_id=0):
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

	def __call__(self, feature_extraction, video, topN = 5):
		for batch_timestamps, batch_frames, extracted_batch_feature in feature_extraction(video):
			feature = extracted_batch_feature.reshape(1,self.batchsize,-1)
			if feature.shape[-1] != self.featureDim:
				feature = self._cut_feature(feature)

			self.mod.forward(Batch([mx.nd.array(feature)]))
			prob = self.mod.get_outputs()[0].asnumpy()
			prob = np.squeeze(prob)
			sorted_idx = np.argsort(prob)[::-1]
			topN_result = OrderedDict()
			for i in sorted_idx[0:topN]:
				topN_result[self.labels[i]] = prob[i]

			yield batch_timestamps, batch_frames, topN_result


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
	from config import config
	import time

	filename = "test.avi"
	frame_group = 5
	video = Video(filename, step=1, frame_group_len=frame_group)
	feature_extract = FeatureExtraction(modelPrototxt=config.FEATURE_EXTRACTION.MODEL_PROTOTXT,
	                                    modelFile=config.FEATURE_EXTRACTION.MODEL_FILE,
	                                    featureLayer=config.FEATURE_EXTRACTION.FEATURE_LAYER, gpu_id=0)
	featurecoding = FeatureCoding(featureDim=config.FEATURE_CODING.FEATURE_DIM, batchsize=frame_group, modelPrefix=config.FEATURE_CODING.MODEL_PREFIX,
	                               modelEpoch=config.FEATURE_CODING.MODEL_EPOCH, synset=config.FEATURE_CODING.SYNSET, gpu_id=0)

	t1 = time.time()
	for timestamps, _, classification_result in featurecoding(feature_extract, video, topN=5):
		t2 = time.time()
		print "time cost: %f, results in timeduration:(%f~%f)s\n"%(t2-t1,timestamps[0],timestamps[-1])
		for label,prob in classification_result.items():
			print "{0}:{1:0.2f}%".format(label,prob*100)

		print "--"*10
		t1 = time.time()


