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
1. 截帧api：      
`video.py`是截帧api的主要脚本，运行demo：`python video.py`。    
其中Video类的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	filename（必须）  | string | 待处理视频名称。 |
	start（可选）  | float | 视频的开始截帧时间点（以秒计），默认从视频的第0秒开始截帧。 |
	end（可选） | float | 视频的结束截帧时间点（以秒计），默认处理到视频的最后。|
	step（可选） | float | 每step秒截帧一次，默认截取每一帧。 |
	verbose（可选）| bool | 在处理视频时显示进度条，默认False。|
	frame\_group\_len (可选)| int | frame\_group\_len帧图片组合成一个输出，相当于batchsize，默认为1。|

2. 特征提取api:    
`featureExtract.py`是特征提取api的主要脚本，它也利用了截帧api。目前支持的特征是SENet。运行demo：`python featureExtract.py`。    
其中FeatureExtract类的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	video（必须）  | class | Video类的object。 |
	modelPrototxt（可选）  | string | 用于特征提取的模型prototxt，默认为'./models/SENet.prototxt'。 |
	modelFile（可选） | string | 模型的caffemodel路径，默认为'./models/SENet.caffemodel'。|
	featureLayer（可选） | string | 进行特征提取的层，默认为'pool5/7x7_s1'。 |
	gpu\_id（可选）| int | 使用gpu id，默认为0。|

	```
	注意:video中设置的frame_group_len需要和model/SENet.prototxt中的batchsize保持一致，否则会报错。
	```

3. 特征融合和多帧分类api：    
`featureCoding.py`是特征融合和多帧分类api的主要脚本，它也利用了截帧api和特征提取api。目前支持的特征融合方法是NetVLAD.运行demo：`python featureCoding.py`。    
其中FeatureCoding类的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	featureDim（可选）  | int | 用于编码的特征维度，需小于输入特征的维度，默认为512维。 |
	batchsize（可选）  | int | 特征编码的batchsize，默认为1。 |
	modelPrefix（可选） | string | 用于特征编码的模型prefix，默认为'models/netvlad'。|
	modelEpoch（可选） | int | 模型epoch，默认为0。 |
	synset（可选）| string | 模型对应labels，默认为'lsvc\_class\_index.txt' |
	gpu\_id（可选）| int | 使用gpu id，默认为0。|

4. 后处理api：
`postProcessing.py`是后处理api的主要脚本，它利用了截帧api，特征提取api，特征融合和多帧分类api，并最后将分类结果做一个整合，输出视频的多个分类标签，及其分别所处的开始时间和结束时间。运行demo：`python postProcessing.py`。    
其中PostProcessing类的参数设置如下：

	参数名  | 类型 | 介绍 |
	------------- | ------------- | -------------|
	score_thresh（必须）  | float | 输出类别的概率阈值，高于此阈值的分类结果可进行整合。 |


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
