# 视频分类inference API （v1）
## Demo
环境配置见下文。配置完成后运行如下代码，可生成视频分类结果的视频。

```
python demo.py --composite_video
```

## 环境配置
1. 起一个mxnet的容器。
2. 进入容器以后，安装下列依赖项：
  ```
  apt-get update
  apt-get install ffmpeg libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
  pip install  tqdm
apt-get install --no-install-recommends libboost-all-dev
pip install protobuf
apt-get install cython python-skimage python-pip
  ```

3.下载senet版caffe（链接问我要）,	并编译：
  ```
  cd $CAFFE_ROOT
  mkdir build
  cd build
  cmake ..
  make all
  make install
  ```

4. 开始使用:    
下载一个测试视频（test.avi），下载senet模型和netvlad模型放到models/目录下（链接问我要）。    
然后运行如下代码，可生成视频分类结果的视频：
```
python demo.py --video_path test.avi --composite_video
```

## API Details
### video_infer:
```
class video_infer
__init__(FeatureExtraction, FeatureCoding, PostProcessing)
[label_durations，video_labels, label_probs] = infer(video)
demo_video = composite_video(video)
```

|参数名|类型|介绍|
|---|---|---|
| FeatureExtraction |类|用于视频帧特征提取|
| FeatureCoding |类|用于视频帧编码|
| PostProcessing |类|用于视频处理结果的融合|
| video |类|用于视频文件的抽象|
| label_durations |list| 视频中动作的开始和结束时间，offset |
| video_labels |list| 视频中动作类别 |
| label_probs |list| 视频中动作的概率 |
| demo_video |string|输出视频的文件名|


### 视频文件api：      
`video.py`是 视频文件的抽象主要用于截帧，运行demo：`python video.py`。

```
class Video
__init__(filename, start, end，step，verbose，frame_group_len)
[timestamps, frames] = Video()
```

其中Video类的对外接口包括初始化和截帧两个部分。  
1. 初始化函数\_\_init\_\_()的参数设置如下：
    
    |参数名  | 类型 | 介绍 |
    |------------- | ------------- | -------------|
    |filename（必须）  | string | 待处理视频名称。 |
    |start（可选）  | float | 视频的开始截帧时间点（以秒计），默认从视频的第0秒开始截帧。 |
    |end（可选） | float | 视频的结束截帧时间点（以秒计），默认处理到视频的最后。|
    |step（可选） | float | 每step秒截帧一次，默认截取每一帧。 |
    |verbose（可选）| bool | 在处理视频时显示进度条，默认False。|
    |frame\_group\_len (可选)| int | frame\_group\_len帧图片组合成一个输出，相当于batchsize，默认为1。注意：所设置的frame\_group\_len需要和特征提取api中modelPrototxt的batchsize保持一致，否则会报错。|
		
2. 截帧需要调用\_\_iter\_\_()函数，此函数无输入参数。

      输出 | 类型 | 介绍 |
      ------------- | ------------- | -------------|
      timestamps| deque（list） | group的一组帧对应的时间戳|
      frames| deque（list） | 帧组|
		
### 特征提取api:    
`featureExtract.py`是特征提取api的主要脚本，它也利用了截帧api。目前支持的特征是SENet。运行demo：`python featureExtract.py`。    #

```
class FeatureExtract
__init__(modelPrototxt, modelFile，featureLayer，gpu_id)
[timestamps, frames，features] = FeatureExtract(video)
```

其中FeatureExtract类的对外接口包括初始化和特征提取两个部分。    
1. 初始化函数\_\_init\_\_()的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	modelPrototxt（可选）  | string | 用于特征提取的模型prototxt，默认为'./models/SENet.prototxt'。注意：modelPrototxt中的batchsize需要和截帧api中设置的frame\_group\_len保持一致，否则会报错。|
	modelFile（可选） | string | 模型的caffemodel路径，默认为'./models/SENet.caffemodel'。|
	featureLayer（可选） | string | 进行特征提取的层，默认为'pool5/7x7_s1'。 |
	gpu\_id（可选）| int | 使用gpu id，默认为0。|
2. 特征提取需要调用\_\_call\_\_()函数，此函数无输入参数。

	输出 | 类型 | 介绍 |
	------------- | ------------- | -------------|
	video（必须）  | class | Video类的object。 |
	timestamps| deque（list） | group的一组帧对应的时间戳。 |
	frames| deque（list） | 帧组。 |
	features | numpy array | 帧组的特征，维度为batchsize*featureDim。 |
		
### 特征融合和多帧分类api：    
`featureCoding.py`是特征融合和多帧分类api的主要脚本，它也利用了截帧api和特征提取api。目前支持的特征融合方法是NetVLAD.运行demo：`python featureCoding.py`。  

```
class FeatureCoding
__init__(featureDim, batchsize, modelPrefix，modelEpoch，gpu_id)
[batch_timestamps, batch_frames, topN_result] = FeatureCoding(feature_extraction，video，topN)
```
其中FeatureCoding类的对外接口包括初始化和特征融合分类两个部分。    
1. 初始化函数\_\_init\_\_()的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	featureDim（可选）  | int | 用于编码的特征维度，需小于输入特征的维度，默认为512维|
	batchsize（可选）  | int | 特征编码的batchsize，默认为1|
	modelPrefix（可选） | string | 用于特征编码的模型prefix，默认为'models/netvlad'|
	modelEpoch（可选） | int | 模型epoch，默认为0|
	synset（可选）| string | 模型对应labels，默认为'lsvc\_class\_index.txt'|
	gpu\_id（可选）| int | 使用gpu id，默认为0|
	
2. 特征融合+分类函数\_\_call\_\_()函数的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	feature_extraction（必须）  | class | FeatureExtract类的object。 |
	topN（可选）  | int | 输出topN的分类结果，默认为5。 |
			
	输出 | 类型 | 介绍 |
	------------- | ------------- | -------------|
	batch_timestamps| deque（list） | group的一组帧对应的时间戳。 |
	batch_frames| deque（list） | 帧组。 |
	topN_result | dict | topN的分类结果，key为类别名，value为类别概率。 |

### 后处理api：
`postProcessing.py`是后处理api的主要脚本，它利用了截帧api，特征提取api，特征融合和多帧分类api，并最后将分类结果做一个整合，输出视频的多个分类标签，及其分别所处的开始时间和结束时间。运行demo：`python postProcessing.py`。    
```
class PostProcessing
__init__(score_thresh)
[label_durations, video_labels，label_probs] = PostProcessing(batch_timestamps，batch_classification_result)
```

其中PostProcessing类的对外接口包括初始化和后处理两个部分。   
1. 初始化函数\_\_init\_\_()的参数设置如下：
	
	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	score_thresh（必须）  | float | 输出类别的概率阈值，高于此阈值的分类结果可进行整合。 |

2. 后处理函数\_\_call\_\_()函数的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	batch_timestamps（必须）| list | 单次视频的时间戳|
	batch_classification_result（必须）  | list | 单次视频分类的结果|
		
	输出 | 类型 | 介绍 |
	------------- | ------------- | -------------|
	video_labels	|list	| 视频中动作类别。|
	label_durations | list	|视频中动作的开始和结束时间，list中的每个item为[start,end]的list。|
	label_probs|	list	|视频中动作的概率。|
## 技术方案
```
输入：input video
-> 截帧 （ffmpeg）
-> 特征提取  （SENet）
-> 多帧特征融合（NetVLAD）
-> 多帧分类（FC）
-> 后处理
输出：视频的多个分类标签，及其分别所处的开始时间和结束时间
```

### 1. 截帧
采用ffmpeg对视频进行处理，包括：
（1）视频信息提取：帧率，帧大小，视频时长等；
（2）截帧：包括逐帧截取，和每n帧截取1帧两种截帧方式；
（3）形成帧组： 为便于后面进行多帧特征融合的方便，在此处将多帧组合进行输出。

### 2. 特征提取
采用SENet对帧进行特征提取，输入不定长的帧，输出不定长的SENet特征。
目前提特征所采用的框架是caffe，代码已经从叶博那边要到了，需要重新建个镜像调试好专门用于提特征。

### 3. 多帧特征融合
采用NetVLAD将多帧特征进行重新编码，输入多帧特征，输出一个新的特征表示。实际中，3和4两个步骤是合在一起训练的，因此inference也在一起。

### 4. 多帧分类
将NetVLAD输出的特征输入到FC中进行视频分类。实际中3和4两个步骤是合在一起的。

### 5. 后处理
将分类结果做一个整合，输出视频的多个分类标签，及其分别所处的开始时间和结束时间。
