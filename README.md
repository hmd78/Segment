# Segment
## Sources
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [MMengine](https://github.com/open-mmlab/mmengine)

>mmdetection and mmcv version should be exactly those written in the notebook
>
>mmdetection 3.x doesnt support all mmdetection main branch so couldn't write code with all python API functions and classes
>
>Instead used mmengine to train and make inference
>
>MMCV version should be more than 2.0.0

### Instance segmentation using RTMDet

Training FashionPedia dataset with 48K training images

Have gotten a subset of data and rewrite an annotation file

Changed config file for training
