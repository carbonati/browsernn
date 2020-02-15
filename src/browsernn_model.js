(function(global){
  "use strict";


  var Model = function() {
    this.layers = [];
  }

  Model.prototype = {
    createLayers: function(defs, seed) {
      // args:
        // defs: list containing each layer and their respective hyper
                //       parameters
        // seed: seed to reinitialize random variables

      // returns:
        // object containing each layer as an instance

      if (defs.length < 2) {
        error_message = 'Error! The model must include at least one ' +
                                'input layer and one loss layer.';
        throw error_message;
      }
      if (defs[0].type != 'input') {
        error_message = 'Error! The first layer of the model must be ' +
                                'an input layer.';
        throw error_message;
      }

      var addLayers = function(layers) {
        // args:
          // layers: list containing each layer defined by the user

        // returns:
          // returns a list containing all user defined layers,
                    // activation functions and any extra layers not required
                    // to be defined by the user

        var all_layers = [];

        for (var i=0; i<layers.length; i++) {
          layer = layers[i]

          if (layer.type == 'regression') {
            all_layers.push({type: 'dense', n_neurons: layer.n_neurons})
           }

           // for any output layer (softmax or SVM) create a dense
                    // network between the previous layer to the output layer
           if (layer.type == 'softmax' || layer.type == 'svm') {
             all_layers.push({type: 'dense', n_neurons: layer.n_classes})
           }

          if ((layer.type == 'dense') && (typeof layer.bias_pref == 'undefined')) {
            layer.bias_pref = 0.0;

            if (typeof layer.activation == 'relu') {
              layer.bias_pref = 0.1;
            }
          }

          all_layers.push(layer);

          if (typeof layer.activation != 'undefined') {
            if (layer.activation == 'relu') {
              all_layers.push({type: 'relu'});
            } else if (layer.activation == 'tanh') {
              all_layers.push({type: 'tanh'});
            } else if (layer.activation == 'sigmoid') {
              all_layers.push({type: 'sigmoid'});
            } else if (layer.activation == 'linear') {
              all_layers.push({type: 'linear'});
            } else if (layer.activation == 'maxout') {
              var gp_size = layer.group_size == 'undefined' ? 2 : layer.group_size;
              all_layers.push({type: 'maxout', group_size: gp_size})
            } else {
              console.log('Error! ' + layer.activation + ' is not supported')
            }
          }

          if ((typeof layer.drop_prob != 'undefined') && (layer.type == 'dropout')) {
            all_layers.push({type: 'dropout', drop_prob: layer.drop_prob})
          }
        }
        return all_layers;
      }

      var layers = addLayers(defs);

      this.layers = [];

      // instantiate each layer from the model
      for (var i=0; i<layers.length; i++) {
        var layer = layers[i];

        // for each layer (after input) pass in the previous layers weights
        if (i > 0) {
          var prev_layer = this.layers[i-1];
          layer.in_n = prev_layer.out_n;
          layer.in_d = prev_layer.out_d;
          layer.in_depth = prev_layer.out_depth;
        }

        switch(layer.type) {
          case 'dense': this.layers.push(new global.DenseLayer(layer, seed)); break;
          case 'input': this.layers.push(new global.InputLayer(layer)); break;
          case 'sigmoid': this.layers.push(new global.SigmoidLayer(layer)); break;
          case 'tanh': this.layers.push(new global.TanhLayer(layer)); break;
          case 'relu': this.layers.push(new global.ReluLayer(layer)); break;
          case 'maxout': this.layers.push(new global.MaxoutLayer(layer)); break;
          case 'softmax': this.layers.push(new global.SoftmaxLayer(layer)); break;
          case 'svm': this.layers.push(new global.SVMLayer(layer)); break;
          case 'linear': this.layers.push(new global.LinearLayer(layer)); break;
          case 'dropout': this.layers.push(new global.DropoutLayer(layer)); break;
          default: console.log('Error! ' + layer.type + ' is not supported');
        }

      }
    },

    // forward propogation
    forward: function(tensor, trainable) {
      // args:
        // tensor:
        // trainable: (boolean) true if training - false if predicting

      if(typeof trainable == 'undefined') trainable = false;
      var units = this.layers[0].forward(tensor, trainable);
      for (var i=1; i<this.layers.length; i++) {
        units = this.layers[i].forward(units, trainable);
      }
      return units;
    },

    // backpropogation
    backward: function(y) {
      // args:
        // y: output (label)

      // returns:
        // loss score at the current state of the model

      var n_layers = this.layers.length;
      var loss = this.layers[n_layers-1].backward(y); // cost from the output layer

      for (var i=n_layers-2; i>=0; i--) {
        this.layers[i].backward();
      }
      return loss;
    },

    getParamsAndGrads: function() {
      // args:
        //

      // returns: a list of objects that each contain the parameters,
      //          gradients and L1/L2 consts among all nodes in each layer
      //          the list should return N neurons + N dense layers + N
      //          classes + 1

      var pg_ls = [];
      for (var i=0; i<this.layers.length; i++) {
        var layer_pg_ls = this.layers[i].getParamsAndGrads();
        for (var j=0; j<layer_pg_ls.length; j++) {
          pg_ls.push(layer_pg_ls[j]);
        }
      }
      return pg_ls;
    }
  }

})(browsernn);
