# Computer Vision Course Projects

This project is about tasks for Computer Vision Course in Electronic Information and Communication Department, Huazhong University of Science and Technology.

## Project navigation
name | usage
:----:|:----:
[Proj1](https://github.com/Kexin-Tang/CV_CourseProject/blob/master/perceptron.py) | perceptron for linear classification
[Proj2](https://github.com/Kexin-Tang/CV_CourseProject/blob/master/neural_network.py) | one layer neural network for classification
[Proj3](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/mnist) | multi-layer neural network for mnist classification
[Proj4](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/pytorch) | pytorch version for mnist or cifar10 classification



### Proj1 - perceptron
<img src="https://i.loli.net/2020/09/23/s1lwqPMGhbjfnHS.png" width = "450" height = "300" alt="perceptron"/><img src="https://i.loli.net/2020/09/27/SrINkUFfJAewBLi.png" width = "450" height = "300" alt="loss"/>

### Proj2 - neural network
<img src="https://i.loli.net/2020/09/27/Ag5c4GEhKy8vtZU.png" width = "450" height = "300" alt="dbmoon"/>

### Proj3 - mnist
file | usage
:----:|:----:
function.py | define the activate function such as ReLU, softmax and sigmoid
Net.py      | define the network structure and forward/backward process 
mnist.py    | how to load data and label from .gz files
main.py     | batch learning and figure plotting

Different activate functions will contribute to different output

<img src="https://i.loli.net/2020/10/05/Law8IhSVxclJjDG.png" width = "1000" height = "300" alt="activate_function"/>

Different init principle will contribute to different output

<img src="https://i.loli.net/2020/10/05/Qrg83Ct5vjehBDZ.png" width = "1000" height = "300" alt="init"/>

### Proj4 - pytorch
file | usage
:----:|:----:
MyNet.py | generate a simply CNN model
ResNet.py      | transfer learning from Res50
VGG.py    | transfer learning from VGG16
Plot.py     | plot the final results, such as loss and acc

*note: In ./pytorch/models I only store three models, because Res50 and VGG16 models are too huge to store in Github*

<img src="https://i.loli.net/2020/10/07/K34BasirfH1WRlD.png" width = "300" height = "200" alt="MyNet_acc"/><img src="https://i.loli.net/2020/10/07/cmzdZabGqHkMwRi.png" width = "300" height = "200" alt="MyNet_loss"/>

<img src="https://i.loli.net/2020/10/13/r9ISqshJxX8Uk45.png" width = "300" height = "200" alt="Res50&VGG16_acc"><img src="https://i.loli.net/2020/10/13/f5UNYhHm7J6TLab.png" width = "300" height = "200" alt="Res50&VGG16_loss">
