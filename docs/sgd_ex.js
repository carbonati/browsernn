var browsernn = require('../build/browsernn.js');

params = {
    optimizer: 'SGD',
    learning_rate: 0.01,
    momentum: 0.9,
    l2_decay: 0.01,
    batch_size: 10
}
trainer = new browsernn.Trainer(model, layers, params);