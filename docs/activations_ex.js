var browsernn = require('../build/browsernn.js');

layers = [];
layers.push({type = 'dense', n_neurons: 64, activation: 'sigmoid'});
layers.push({type = 'dense', n_neurons: 64, activation: 'relu'});
layers.push({type = 'dense', n_neurons: 64, activation: 'tanh'});
layers.push({type = 'dense', n_neurons: 64, activation: 'linear'});