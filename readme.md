# LGANet: Lightweight Joint Global Attention Network for RGB-D Camouflaged Object Detection

Source code and dataset for our paper "LGANet: Lightweight Joint Global Attention Network for RGB-D Camouflaged Object Detection" by Yang Liu, Shuhan Chen, Haonan Tang, and Shiyu Wang. 
The code will be uploaded subsequently.

# Training/Testing

The training and testing experiments are conducted using [PyTorch]( https://github.com/pytorch/ ) with one NVIDIA 3090 GPU of 24 GB Memory.

1.  Configuring your environment (Prerequisites):

  * Python 3.7+, Pytorch 1.5.0+, Cuda 10.2+, TensorboardX 2.1, opencv-python <br>
      If anything goes wrong with the environment, please check requirements.txt for details.

2.  Downloading necessary data:

  * New depth map datasets are being uploaded.

# Evaluation
                                                                          
1.  CODToolbox：（ https://github.com/DengPingFan/CODToolbox ）- By DengPingFan(<https://github.com/DengPingFan>)

2.  Precision_and_recall：（ https://en.wikipedia.org/wiki/Precision_and_recall ）  
 
3.  Dice/IoU_Measure：cited from the paper titled 'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation' by Milletari, Fausto.

# Network hyperparameters:

The epoch size and batch size are set to 100 and 10, respectively.
The PyTorch library was used to implement and train our model, which was trained using Adam optimization,
regularization was conducted using a weight decay of 1e-3, and we set the learning rate of our training phase to 1e-4.

# Pretraining Models

  * Pre_Model.zip：https://pan.baidu.com/s/1ygJU-GH3OEhx7XqsMpoV9Q 提取码: 353X 

# Reproduce

1.  Network training

  * python train.py   

  * parser.add_argument('--train_root', type=str, default='', help='the train images root')
    parser.add_argument('--val_root', type=str, default='', help='the val images root')
    parser.add_argument('--save_path', type=str, default='/', help='the path to save models and logs')

2.  Network testing

  * python test.py   

  * parser.add_argument('--test_path',type=str,default='/',help='test dataset path')

  * model.load_state_dict(torch.load('/Net_epoch_best.pth', weights_only=True))
    model_path = '/Net_epoch_best.pth'

  * save_path = '/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = '/' + dataset + '/'
    if not os.path.exists(edge_save_path):
        os.makedirs(edge_save_path)

3.   Evaluation of quantitative indicators

 *   python CODtest_metrics.py

#  Architecture and Details

![1](https://github.com/YangLiu353/LGA/blob/3400a8808e433d784efe7a5404fb798e637aca60/1.jpg?raw=true)
# Results
![2](https://github.com/YangLiu353/LGA/blob/4772f139f1d808194876d187cc2a00c7497549b3/2.jpg?raw=true)
![3](https://github.com/YangLiu353/LGA/blob/4772f139f1d808194876d187cc2a00c7497549b3/3.jpg?raw=true)
