# Computer Vision Course Projects

This project is about tasks for Computer Vision Course in Electronic Information and Communication Department, Huazhong University of Science and Technology.

## Project navigation
name | usage
:----:|:----:
[Proj1](https://github.com/Kexin-Tang/CV_CourseProject/blob/master/perceptron.py) | perceptron for linear classification
[Proj2](https://github.com/Kexin-Tang/CV_CourseProject/blob/master/neural_network.py) | one layer neural network for classification
[Proj3](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/mnist) | multi-layer neural network for mnist classification
[Proj4](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/pytorch) | pytorch version for mnist or cifar10 classification
[Proj5](https://github.com/Kexin-Tang/CV_CourseProject/tree/master/Localization) | pytorch version for single object localization



### Proj1 - perceptron
<img src="https://i.loli.net/2020/09/23/s1lwqPMGhbjfnHS.png" width = "300" height = "200" alt="perceptron"/><img src="https://i.loli.net/2020/09/27/SrINkUFfJAewBLi.png" width = "300" height = "200" alt="loss"/>

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

### Proj4 - pytorch classification
file | usage
:----:|:----:
MyNet.py | generate a simply CNN model
ResNet.py      | transfer learning from Res50
VGG.py    | transfer learning from VGG16
Plot.py     | plot the final results, such as loss and acc

*note: In ./pytorch/models I only store three models, because Res50 and VGG16 models are too huge to store in Github*

<img src="https://i.loli.net/2020/10/07/K34BasirfH1WRlD.png" width = "300" height = "200" alt="MyNet_acc"/><img src="https://i.loli.net/2020/10/07/cmzdZabGqHkMwRi.png" width = "300" height = "200" alt="MyNet_loss"/>

<img src="https://i.loli.net/2020/10/13/r9ISqshJxX8Uk45.png" width = "300" height = "200" alt="Res50&VGG16_acc"><img src="https://i.loli.net/2020/10/13/f5UNYhHm7J6TLab.png" width = "300" height = "200" alt="Res50&VGG16_loss">

### Proj5 - pytorch single localization
file | usage
:----:|:----:
Net.py | VGG16 pretrained network + customize
VGG_loss_weight.py | main file, use two loss funcs (cross entropy and Smooth L1) 
VGGConV_xxx.py    | using ConV instead fc
visualization.py     | plot the final results
dataloader.py     | load data from file as ndarray and tensor
tiny_vid.tar    |   dataset sampled from VID

<img src="https://i.loli.net/2020/10/15/Pt7npfwHGxdgUQi.png" width = "300" height = "200" alt="weight=1:1"><img src="https://i.loli.net/2020/10/15/cYXB9abdMRvhewP.png" width = "300" height = "200" alt="weight=1:1e-1">

<img src="https://i.loli.net/2020/10/15/COuy3rHoIsi5X4U.png" width = "300" height = "200" alt="weight=1:2e-2"><img src="https://i.loli.net/2020/10/15/qzSriCcQGx5bedM.png" width = "300" height = "200" alt="ConV">
