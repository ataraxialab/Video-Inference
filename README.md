# 视频分类inference API （v1）
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

## 使用姿势 （截帧+特征提取）
1. 起一个caffe的容器。在k8s上image设置为`reg-xs.qiniu.io/atlab/atnet-caffe-trainer:20170714v1`
2. 进入容器以后，安装下列依赖项：

  apt-get update
  apt-get install ffmpeg
  pip install  tqdm

3. 重新编译caffe：
下载senet版caffe：（链接问我要）。
解压到/opt/caffe/目录下（将目录下原有的caffe删除）
然后就是正常caffe的编译流程了：

  mkdir build
  cd build
  cmake ..
  make all
  make install

4. 开始使用截帧api：
`video.py`是截帧api的主要脚本，可以仿造main函数设置filename运行即可。
对video可设置的选项：

	start : float, optional
			Begin iterating frames at time `start` (in seconds).
			Defaults to 0.
	end : float, optional
			Stop iterating frames at time `end` (in seconds).
			Defaults to video duration.
	step : float, optional
			Iterate frames every `step` seconds.
			Defaults to iterating every frame.
	verbose : bool, optional
			Show a progress bar while iterating the video. Defaults to False.
	frame_group_len : int
			Number of frames to be grouped as an output. Defaults to 1.

5. 开始使用特征提取api:
`featureExtract.py`是特征提取api的主要脚本，它也利用了截帧api。可以仿造main函数设置filename进行运行。
目前支持的特征是SENet（模型问我要，然后放到model/目录下）。需要注意的是video中设置的frame\_group\_len需要和model/SENet.prototxt中的batchsize一致，否则会报错。