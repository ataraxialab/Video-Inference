from easydict import EasyDict as edict

config = edict()

config.FEATURE_EXTRACTION = edict()
config.FEATURE_EXTRACTION.MODEL_PROTOTXT = 'models/SENet.prototxt'
config.FEATURE_EXTRACTION.MODEL_FILE = 'models/SENet.caffemodel'
config.FEATURE_EXTRACTION.FEATURE_LAYER = 'pool5/7x7_s1'

config.FEATURE_CODING = edict()
config.FEATURE_CODING.MODEL_PREFIX = 'models/netvlad'
config.FEATURE_CODING.MODEL_EPOCH = 50
config.FEATURE_CODING.FEATURE_DIM = 512
config.FEATURE_CODING.SYNSET='lsvc_class_index.txt'