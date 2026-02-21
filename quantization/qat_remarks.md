# remarks on qat training for hardware deployment (lu: 11 Feb 2026)

* Cannot use nn.tanh/qnn.quanttanh as FINN is unable to convert these layers to hw
* Use quanthardtanh instead, which is translated to regular blocks rather than specialised tanh blocks
* quanthardtanh limits set to max=1.0 and min=-1.0, may lead to empty/unused autograds
* hence, must set autograd to output zeros when none
* leads to reduced accuracy compared to normal tanh due to loss of gradients
