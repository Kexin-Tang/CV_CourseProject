# Computer Vision Course Projects

## Project navigation
name | usage
:----:|:----:
[Proj1](https://github.com/Kexin-Tang/CV_CourseProject/blob/master/perceptron.py) | perceptron for linear classification
[Proj2](https://github.com/Kexin-Tang/CV_CourseProject/blob/master/neural_network.py) | one layer neural network for classification
[Proj3](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/mnist) | multi-layer neural network for mnist classification
[Proj4](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/pytorch) | pytorch version for mnist or cifar10 classification



#### Proj1 - perceptron
![perceptron.png](https://i.loli.net/2020/09/23/s1lwqPMGhbjfnHS.png) ![loss.png](https://i.loli.net/2020/09/27/SrINkUFfJAewBLi.png)

#### Proj2 - neural network
![dbmoon.png](https://i.loli.net/2020/09/27/Ag5c4GEhKy8vtZU.png)

#### Proj3 - mnist
file | usage
:----:|:----:
function.py | define the activate function such as ReLU, softmax and sigmoid
Net.py      | define the network structure and forward/backward process 
mnist.py    | how to load data and label from .gz files
main.py     | batch learning and figure plotting

Different activate functions will contribute to different output
![activate_function.png](https://i.loli.net/2020/10/05/Law8IhSVxclJjDG.png)

Different init principle will contribute to different output
![init.png](https://i.loli.net/2020/10/05/Qrg83Ct5vjehBDZ.png)

#### Proj4 - pytorch
file | usage
:----:|:----:
MyNet.py | generate a simply CNN model
ResNet.py      | transfer learning from Res50
VGG.py    | transfer learning from VGG16
Plot.py     | plot the final results, such as loss and acc

![MyNet_acc.png](https://i.loli.net/2020/10/07/K34BasirfH1WRlD.png)
![MyNet_loss.png](https://i.loli.net/2020/10/07/cmzdZabGqHkMwRi.png)
