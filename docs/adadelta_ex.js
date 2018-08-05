var browsernn = require('../build/browsernn.js');

params = {
    optimizer: 'adadelta',
    learning_rate: 1.0,
    epsilon: 1e-7,
    rho: 0.95,
    batch_size: 10
}
trainer = new browsernn.Trainer(model, layers, params);