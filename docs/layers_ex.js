var browsernn = require('../build/browsernn.js');

layers = [];

// multilayer perceptron
layers.push({type: 'input', in_n: 1, in_d: 2, in_depth: 1});
layers.push({type: 'dense', n_neurons: 8, activation: 'relu'});
layers.push({type: 'dense', n_neurons: 4, activation: 'relu'});
layers.push({type: 'softmax', n_classes: 2});