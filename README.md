# browserNN.js

browserNN is a javascript library that allows you to train deep neural networks in your web browser. Inpsired by [@karpathy](https://github.com/karpathy).

For full documentation and a demo, please see: https://carbonati.github.io/posts/browsernn/

## Example
```javascript
model = new browsernn.Model();

layers = []; 

layers.push({type: 'input', n: 1, d: 2, depth: 1}); 
layers.push({type: 'dense', n_neurons: 8, activation: 'relu'}); 
layers.push({type: 'dense', n_neurons: 4, activation: 'tanh'}); 
layers.push({type: 'softmax', n_classes: 2}); 

params = {
    optimizer: 'sgd', 
    learning_rate: .01, 
    momentum: 0.01, 
    batch_size: 32,
    l2_decay: 0.01
};
        
trainer = new browsernn.Trainer(model, layers, params); 
```

## Use in Node
1. install: `npm install browsernn`
2. Use: `var browsernn = require("browsernn");`

## License
MIT
