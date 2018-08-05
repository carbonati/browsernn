var browsernn = require('../build/browsernn.js');

params = {
    optimizer: 'adagrad',
    learning_rate: 0.01,
    epsilon: 1e-7,
    batch_size: 10
}
trainer = new browsernn.Trainer(model, layers, params);