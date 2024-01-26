# StereoNet for satellite
This is a non-official pytorch implementation of stereo matching network [StereoNet](https://arxiv.org/pdf/1807.08865.pdf) and make a slight modification to adapt it to satellite image
### Differences from original paper
- In order to make the model applicable to satellite images, the option of minimum disparity was added to adapt it to negative disparity
- Remove the last ReLU that constrain the disparity to be positive in the EdgeAwareRefinement module 
- Using OneCycleLR to adjust the learning rate

### Dataset
The dataloader only support `DFC2019` and `WHU-Stereo` dataset<br />
The dataloader can support both 3channel 8bit and 1channel 16bit image. <br />
You can train the model by using `train.py`

```
python train.py
```
You can use tensorboard to monitoring the loss and learning rate during training.

```
tensorboard --logdir ./logs
```

### Environment
- torch                     2.1.1
- torchvision               0.16.1
- numpy                     1.24.1
- matplotlib                3.2.2
- opencv-python             4.8.1.78

### demo
You can test a single pair of stereo image using the notebook `demo.ipynb`<br />
Demo is based on 3 channel model. If your trained model is train on single channel image, you need to modify the notebook.