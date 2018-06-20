### Implement LSUV

#### ICLR 2016 paper

- original paper url : https://arxiv.org/abs/1511.06422
- ALL YOU NEED IS A GOOD INIT(Dmytro Mishkin, Jiri Matas)

#### Usage

```
from keras.applications.vgg16 import VGG16
from lsuv import LSUV

model = LSUV(model, x_data[:batch_size])
}
```


#### Explanation

LSUV : Layer-sequential unit-variance

LSUV는 학습을 진행하기 전 첫번 째 mini-batch의 batch normalization
로 해석할 수 있다.

Batch normalization과의 비교실험에서 동등하거나 약간의 성능향상이 있었지만 차이는 critical 하지 않다.
또한 ImageNet 의 Data에서 초기 convergence는 좋았으나 결국 어떠한 큰 차이는 없다는 점.
LSUV가 특정 initializer(orthonormal metric)에서 잘 상응한다는 점에서 자주 사용할 지는 의문.

하지만 Batch Normalization과 달리 첫번째 mini-batch에만 적용되어 계산되기 때문에 computation complexity
는 효율적일 수 있다. 