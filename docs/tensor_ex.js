var browsernn = require('../build/browsernn.js');

tensor = new browsernn.Tensor(1, 2, 1, 0.0); // 1x2x1 tensor initialized with zeros
tensor = new browsernn.Tensor(28, 28, 3); // 28x28x3 tensor with randomly initialized values

var data = [[10,20,30],[40,50,60]];
tensor = new browsernn.Tensor(2, 3, 1, data); // 2x3x1 tensor initialized with data values

tensor.w[4]; // returns 50
tensor.dw[4]; // returns 0 since no derivative has been computed yet