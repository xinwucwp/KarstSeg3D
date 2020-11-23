# KarstSeg3D: Deep Learning for Characterizing Paleokarst Collapse Features in 3-D Seismic Images

**This is a [Keras](https://keras.io/) version of KarstSeg implemented by [Xinming Wu](http://www.jsg.utexas.edu/wu/) for Paleokarst segmentation in 3D seismic images**

As described in **Deep Learning for Characterizing Paleokarst Collapse Features in 
3-D Seismic Images** by [Xinming Wu](http://cig.ustc.edu.cn/xinming/list.htm)<sup>1</sup>, 
[Shangsheng Yan](http://cig.ustc.edu.cn/shangsheng/list.htm)<sup>1</sup>, 
[Jie Qi](https://scholar.google.com/citations?user=p3dQEsIAAAAJ&hl=en)<sup>2</sup> and 
[Hongliu Zeng](https://www.beg.utexas.edu/people/hongliu-zeng)<sup>3</sup>.
<sup>1</sup>[Computational Interpretation Group](http://cig.ustc.edu.cn/), USTC;<sup>3</sup>
The University of Oklahoma; <sup>3</sup>BEG, UT Austin.

## Getting Started with Example Model for paleokarst prediction

If you would just like to try out a pretrained example model, 
then you can download the [pretrained model](https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd) and use the `<apply.py>` script to run a demo. 
I recommend to run the prediction on CPU `<./cpurun apply.py>`, which 
is fast enough.

### Dataset

**To train our CNN network, we automatically created 120 pairs 
of synthetic seismic and corresponding karst volumes, which were 
shown to be sufficient to train a good karst segmentation network.** 

**The training and validation datasets can be downloaded [here](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8)**

### Training

Run `<train.py>` to start training a new karstSeg model by using the 120 synthetic datasets

## Publications

If you find this work helpful in your research, please cite:
    @article{wu2020karstSeg,
        author = {Xinming Wu and Shangsheng Yan and Jie Qi and Hongliu Zeng},
        title = {Deep Learning for Characterizing Paleokarst Collapse Features in 3-{D} Seismic Images 3{D}},
        journal = {Journal of Geophysical Research: Solid Earth},
        volume = {125},
        number = { },
        doi = {doi.org/10.1029/2020JB019685},
        year = {2020},
    }

## License

This extension to the Keras library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
