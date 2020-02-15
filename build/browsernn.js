// things to add:
	// weight initializations - https://keras.io/initializers/
	// epochs - the network depends on the the number of times .fit() is called
	// RMSprop
	// recurrent networks
	// convolutional networks
	// more activation/noise/normalization layers - https://keras.io/activations/
	// attention

// potential bugs:
	// setting random seed converges slower

var browsernn = browsernn || {VERSION: '1.0.0'};
(function(global) {
	"use strict";

	function zeros(n) {
		if(typeof ArrayBuffer === 'undefined') {
	    	var arr = new Array(n);
	      	for(var i=0;i<n;i++) { arr[i]= 0; }
	      	return arr;
	    } else {
	    	  return new Float64Array(n);
	    }
	}

	// generates random value set by seed
	function random(seed) {
		if (typeof seed != "undefined") {
		    var x = Math.sin(seed++) * 10000;
		    return x - Math.floor(x);
		} else {
			return Math.random();
		}
	}

	// generate a random real/integer/natural with some given parameters
	var randr = function(a,b,seed) { return random(seed)*(b-a)+a; }
	var randi = function(a,b,seed) { return Math.floor(random(seed)*(b-a)+a); }

	var randGauss = function(seed) {
		// args:
			// seed: seed to reinitialize random variables

		// returns:
			// gaussian random variable

		var u = typeof seed != 'undefined' ? 2*random(seed)-1 : 2*Math.random()-1
		var v = typeof seed != 'undefined' ? 2*random(seed)-1 : 2*Math.random()-1
		var r = u*u + v*v;
		if(r == 0 || r > 1) {
			if (typeof seed != 'undefined') seed += 1
			return randGauss(seed);
		}
		var c = Math.sqrt(-2*Math.log(r)/r);
	    return u*c;
	}

	var randn = function(mu,sigma,seed) {
		// args:
			// mu: mean
			// sigma: standard deviation
			// seed: seed to reinitilize random variables

		// returns:
			// random real value sampled from mean mu and standard deviation sigma

		return mu+randGauss(seed)*sigma
	}

	// flattens tensors and matrices into 1-d arrays
	function flatten(arr) {
		return arr.reduce(function (flat, toFlatten) {
			return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
		}, []);
	}

	function tanh(x) {
		return (Math.exp(2*x) - 1) / (Math.exp(2*x) + 1);
	}


	var Tensor = function(n, d, depth, init_weight, seed) {
		// args:
			// n: number of rows
			// d: number of columns
			// depth: depth of the tensor
				// if only 2 dimensions are passed as (n,d) the Tensor will create a depth of 1 as Tensor(1, 2, 1)
			// init_weight: array or constant to set the tensors weights to
			// seed: seed to reinitialize random variables

		// returns:
			// tensor with initialized weights of size (n, d, depth)

		this.n = n;
		this.d = d;
		depth = typeof depth != 'undefined' ? depth : 1;
		this.depth = depth;
		this.n_cells = n * d * depth;
		var n_cells = n * d * depth;

		this.w = zeros(this.n_cells);
		this.dw = zeros(this.n_cells);

		// weight normalized parameterization - to reduce variance over the distribution of weights
		if(typeof init_weight == 'undefined') {
			var std = Math.sqrt(1.0/(this.n_cells));
			for (var i=0; i<this.n_cells; i++) {
				// hack for initializing weights that differ between layers
				var reseed = seed+this.n_cells+i;
				this.w[i] = typeof seed != "undefined" ? randn(0.0, std, reseed) : randn(0.0, std, undefined)
			}
		} else if (Object.prototype.toString.call(init_weight) == '[object Array]') {
			var weights_flat = flatten(init_weight);
			for (var i=0; i<n_cells; i++) {
				this.w[i] = weights_flat[i];
			}
		} else {
			for (var i=0; i<n_cells; i++) {
				this.w[i] = init_weight;
			}
		}

	}


	var InputLayer = function(params) {
		// args;
			// params: object containing the dimensions for the input layer of
			// 				 the model.
			// 			   in the case of 2 dimension data like (x_1, x_2) the Tensor
			//				 should take the shape Tensor(1, 2, 1).

		// returns:
			// instantiates the models input layer
		var params = params || {};
		if((params.depth < 1) || (typeof params.depth == 'undefined')) {
			params.depth = 1;
		this.out_n = params.n;
		this.out_d = params.d;
		this.out_depth = params.depth;
		this.layer_type = 'input';
	}

	InputLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor holding the input data (X).
				// trainable: (boolean) whether the model is in train mode.

			// returns:
				// output of the input layer.
			this.input = tensor;
			this.output = tensor;
			return this.output;
		},
		backward: function() {
			return []; // pass
		},
		getParamsAndGrads: function() {
			return []; // pass
		}
	}

	var DenseLayer = function(params, seed) {
		// args:
			// params: hyper parameters of the dense layer that must include the
			//      `  number of neurons and activation function.
			// seed: seed to reinitialize random variables.

		// returns:
			// instantiated dense layer.
		var params = params || {};
		this.n_neurons = params.n_neurons;

        if (typeof params.l1_decay_mul != 'undefined') {
            this.l1_decay_mul = 0.0;
        } else {
            this.l1_deca_mul = params.l1_decay_mul;
        }

        if (typeof params.l2_decay_mul != 'undefined') {
            this.l2_decay_mul = 1.0;
        } else {
            this.l2_deca_mul = params.l2_decay_mul;
        }

		this.n_inputs = params.in_n * params.in_d * params.in_depth;
		this.out_n = 1;
		this.out_d = params.n_neurons;
		this.out_depth = 1;
		this.layer_type = 'dense';

		var bias = typeof params.bias_pref == 'undefined' ? 0.0 : params.bias_pref;
		this.weight_mat = [];
		for (var i=0; i<this.n_neurons; i++) {
			this.weight_mat.push(new Tensor(1, this.n_inputs, 1, undefined, seed));
		}
		this.biases = new Tensor(1, this.n_neurons, 1, bias);
	}

	DenseLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor from the previous layer
				// trainable: (boolean) true if training - false if predicting

			// returns:
				// linear transformation between neurons from the previous layer
				// and the weights form the dense layer
			this.in_units = tensor;
			var output = new Tensor(1, this.n_neurons, 1, 0.0);
			var x = tensor.w;

			for (var i=0; i<this.n_neurons; i++) {
				var linear_sum = 0.0;
				var w_i = this.weight_mat[i].w;
				for (var j=0; j<this.n_inputs; j++) {
					linear_sum += w_i[j] * x[j];
				}
				linear_sum += this.biases.w[i];
				output.w[i] = linear_sum;
			}
			this.out_units = output;
			return this.out_units;
		},

		backward: function() {
			// args:

			// returns:
				// computes derivatives and propagates back through the network
			var input = this.in_units;
			input.dw = global.zeros(input.w.length); //
			// compute gradiet wrt weights & data
			for (var i=0; i<this.n_neurons; i++) {
				var weight_vec = this.weight_mat[i];
				var chain_grad = this.out_units.dw[i]; // grad from prev layer
				for (var j=0; j<this.n_inputs; j++) {
					input.dw[j] += chain_grad * weight_vec.w[j]; // gradwrt inputs
					weight_vec.dw[j] += chain_grad * input.w[j]; // grad wrt weights
				}
			this.biases.dw[i] += chain_grad;
			}
		},
		getParamsAndGrads: function() {
			// args:

			// returns:
				// list containing the layers weights, derivatives, and regularized
				// multipliers.
			var pg_ls = [];
			for (var i=0; i<this.n_neurons; i++) {
				pg_ls.push({
					params: this.weight_mat[i].w,
					grads: this.weight_mat[i].dw,
					l1_decay_mul: this.l1_decay_mul,
					l2_decay_mul: this.l2_decay_mul
				});
			}
			pg_ls.push({
                params: this.biases.w,
                grads: this.biases.dw,
                l1_decay_mul: this.l1_decay_mul,
                l2_decay_mul: this.l2_decay_mul
            });
			return pg_ls;
		}
	}

	var LinearLayer = function(params) {
		// args
			// params: hyper parameters from the previous layer

		// returns:
			// instaniated linear layer
		var params = params || {}
		this.out_n = params.in_n;
		this.out_d = params.in_d;
		this.out_depth = params.in_depth;
		this.layer_type = 'linear'
	}

	LinearLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor holding data from the previous layer (z).
				// trainable: (boolean) whether the model is in train mode.

			// returns:
				// output tensor from the previous layer - since the dense
				// computes a linear transformation.
			this.in_units = tensor;
			this.out_units = tensor;
			return tensor;
		},
		backward: function() {
			return [];
		},
		getParamsAndGrads: function() {
			return [];
		}
	}

	var SoftmaxLayer = function(params) {
		// args:
			// params: hyper parameters from the previous (dense) layer

		// returns:
			// instantiated softmax layer
		var params = params || {};
		// when this function is called we will create a DenseLayer as well to
    // connect the previous layer,
		// then compute the softmax
		this.n_inputs = params.in_n * params.in_d * params.in_depth;
		this.out_n = 1;
		this.out_d = this.n_inputs // out_d == n_classes always
		this.out_depth = 1;
		this.layer_type = 'softmax';
	}

	SoftmaxLayer.prototype = {
		forward: function(tensor, trainable) {
			// args
				// tensor: tensor holding data from the previous (dense) layer
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// output tensor after applying a softmax activation function

			this.in_units = tensor;
			var output = new Tensor(1, this.n_inputs, 1, 0.0);
			// the softmax becomes saturated when the inputs are extremely
      // positive or negative, also becomes saturated when the difference
      // beween inputs is extremely large invariant to adding some scalar
      // value (i.e. the maximum input).
			var z = tensor.w;
			var z_max = tensor.w[0];
			for (var i=1; i<this.out_d; i++) {
				if(z[i] > z_max) z_max = z[i];
			}
			var e_vec = global.zeros(this.out_d);
			var e_sum = 0.0;
			for (var i=0; i<this.out_d; i++) {
				var e = Math.exp(z[i] - z_max);
				e_sum += e;
				e_vec[i] = e;
			}

			// normalize to create a probability distribution
			for (var i=0; i<this.n_inputs; i++) {
				e_vec[i] /= e_sum;
				output.w[i] = e_vec[i];
			}
			this.e_vec = e_vec;
			this.out_units = output;
			return this.out_units;
		},

		backward: function(y) {
			// args:
				// y: softmax prediction

			// returns:
				// negative log likelihood of the predicted class
			var x = this.in_units;
			x.dw = global.zeros(x.w.length);
			for (var i=0; i<this.n_inputs; i++) {
				var target = i == y ? 1.0 : 0.0;
				var l_grad = this.out_units.w[i] - target;
				x.dw[i] = l_grad;
			}
			var loss = -Math.log(this.e_vec[y]);
			return loss;
		},
		getParamsAndGrads: function() {
			return [];
		}
	}

	var SigmoidLayer = function(params) {
		// args:
			// params: hyper parameters from the previous (dense) layer

		// returns:
			// instantiated sigmoid layer

		var params = params || {};
		this.out_n = params.in_n;
		this.out_d = params.in_d;
		this.out_depth = params.in_depth;
		this.layer_type = 'sigmoid';

	}

	SigmoidLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor holding data from the previous layer (z)
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// output tensor after applying element-wise sigmoid function
			this.in_units = tensor;
			var output = new Tensor(this.out_n, this.out_d, this.out_depth, 0.0);
			var N = tensor.w.length;

			for (var i=0; i<N; i++) {
				output.w[i] = 1.0 / (1.0 + Math.exp(-tensor.w[i]));
			}

			this.out_units = output;
			return this.out_units;

		},

		backward: function() {
			// args:

			// returns:
				// computes derivatives and propagates back through the network
			var z = this.in_units;
			var h = this.out_units;
			var N = z.w.length;
			z.dw = global.zeros(N);
			// sigma * (1 - sigma)
			for (var i=0; i<N; i++) {
				z.dw[i] = h.w[i] * (1.0 - h.w[i]) * h.dw[i];
			}
			console.log(z);
		},
		getParamsAndGrads: function() {
			return [];
		}
	}

	var TanhLayer = function(params) {
		// args
			// params: hyper parameters from the previous layer

		// returns:
			// instaniated tanh layer
		var params = params || {};
		this.out_n = params.in_n;
		this.out_d = params.in_d;
		this.out_depth = params.in_depth;
		this.layer_type = 'tanh';
	}

	TanhLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor holding data from the previous layer (z)
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// element-wise Tanh on each unit of the previous layer
			this.in_units = tensor;
			var output = new Tensor(this.out_n, this.out_d, this.out_depth, 0.0);
			var N = tensor.w.length;
			// element-wise tanh
			for (var i=0; i<N; i++) {
				output.w[i] = global.tanh(tensor.w[i]);
			}
			this.out_units = output;
			return this.out_units;
		},

		backward: function() {
			// args:

			// returns:
				// computes derivatives and propagates back through the network
			var z = this.in_units;
			var h = this.out_units;
			var N = z.w.length;
			z.dw = global.zeros(N);
			for (var i=0; i<N; i++) {
				var h_i = h.w[i];
				z.dw[i] = (1.0 - (h_i * h_i)) * h.dw[i];
			}
		},
		getParamsAndGrads: function() {
			return [];
		}
	}

	var ReluLayer = function(params) {
		// args:
			// params: hyper parameters from the the previous layer

		// returns:
			// instantiated ReLu layer
		var params = params || {};
		this.out_n = params.in_n;
		this.out_d = params.in_d;
		this.out_depth = params.in_depth;
		this.layer_type = 'relu';
	}

	ReluLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor holding the input data (X)
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// element-wise ReLu on each each unit of the previous layer
			this.in_units = tensor;
			var output = new Tensor(this.out_n, this.out_d, this.out_depth, 0.0);
			var N = tensor.w.length;
			for (var i=0; i<N; i++) {
				if(tensor.w[i] > 0) output.w[i] = tensor.w[i];
				else output.w[i] = 0;
			}
			this.out_units = output;
			return this.out_units;
		},

		backward: function() {
			// args:

			// returns:
				// computes derivatives and propagates back through the network
			var z = this.in_units;
			var h = this.out_units;
			var N = z.w.length;
			z.dw = global.zeros(N);
			for (var i=0; i<N; i++) {
				if (h.w[i] <= 0) z.dw[i] = 0;
				else z.dw[i] = h.dw[i];
			}
		},
		getParamsAndGrads: function() {
			return [];
		}
	}

	var DropoutLayer = function(params) {
		// args:
			// params: hyper parameters from the previous layer

		// returns:
			// instantiated dropout layer
		var params = params || {};
		this.out_n = params.in_n;
		this.out_d = params.in_d;
		this.out_depth = params.in_depth;
		// default dropout probability to 0.5
		this.drop_prob = typeof params.drop_prob != 'undefined' ? params.drop_prob : 0.5;
		this.dropped = zeros(this.out_n * this.out_d * this.out_depth);
		this.layer_type = 'dropout';
	}

	DropoutLayer.prototype = {
		forward: function(tensor, trainable, seed) {
			// args:
				// tensor: tensor holding data from the previous (activation) layer
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// tensor after applying dropout to the previousl layers weights
			this.in_units = tensor;
			if(typeof trainable == 'undefined') trainable = false;

			var output = new Tensor(this.out_n, this.out_d, this.out_depth, 0.0);
			var N = tensor.w.length;
			if(trainable) {
				for (var i=0; i<N; i++) {
					if (Math.random(seed) < this.drop_prob) {
						output.w[i] = 0;
						this.dropped[i] = true
					} else {
						output.w[i] = this.in_units.w[i];
						this.dropped[i] = false
					}
				}
			} else {
				for (var i=0; i<N; i++) {
					// scaling activations when predicting
					output.w[i] = this.in_units.w[i] * this.drop_prob;

				}
			}
			this.out_units = output;
			return this.out_units;
		},

		backward: function() {
			var h = this.in_units;
			var chain_grad = this.out_units;
			var N = h.w.length;
			h.dw = zeros(N);
			for (var i=0; i<N; i++) {
				// if the value was not dropped pass in the gradient from the previous layer
				// otherwise leave the gradient 0'd out
				if(!(this.dropped[i])) {
					h.dw[i] = chain_grad.dw[i];
				}
			}
		},
		getParamsAndGrads: function() {
			return [];
		}
	}


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
      //		      gradients and L1/L2 consts among all nodes in each layer
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


	var Trainer = function(model, layers, params) {
		// args:
			// model: initialized browsernn model
			// layers: list containing all the layers in the network
			// params: object containing all the hyperparameters

		// returns:
			// configures the model for training and returns:

		this.model = model;
		var params = params || {};
		this.learning_rate = typeof params.learning_rate == 'undefined' ? 0.01 : params.learning_rate;
		this.l1_decay = typeof params.l1_decay == 'undefined' ? 0.0 : params.l1_decay;
		this.l2_decay = typeof params.l2_decay == 'undefined' ? 0.0 : params.l2_decay;
		this.batch_size = typeof params.batch_size == 'undefined' ? 1 : params.batch_size;
		this.optimizer = typeof params.optimizer == 'undefined' ? 'SGD' : params.optimizer;
		this.momentum = typeof params.momentum == 'undefined' ? 0.9 : params.momentum;
		this.ro = typeof params.ro == 'undefined' ? 0.95 : params.ro;
		this.epsilon = typeof params.epsilon == 'undefined' ? 1e-7 : params.epsilon;


		this.seed = typeof params.seed == 'undefined' ? undefined : params.seed;

		this.iter = 0;
		this.g_sum = []; // for momentum
		this.x_sum = []; // for adagrad / adadelta

		model.createLayers(layers, this.seed);

	}

	Trainer.prototype = {
		fit: function(x, y) {
			// args:
				// x: input data
				// y: output (label)

			// returns:
				// Trains the model for 1 full epoch then returns an object
				// containing meta data with the loss and cost metrics

			var start_dt = new Date().getTime();
			this.model.forward(x, true); // propagate the network forward
			var end_dt = new Date().getTime();
			var forward_time = end_dt - start_dt


			var start_dt = new Date().getTime();

			var cost_loss = this.model.backward(y); // propagate the network backward
			var l1_decay_loss = 0.0;
			var l2_decay_loss = 0.0;
			var end_dt = new Date().getTime();
			var backward_time = end_dt - start_dt;

			this.iter++;
			if (this.iter % this.batch_size == 0) {
				var pg_ls = this.model.getParamsAndGrads();

				if ((this.optimizer != 'SGD' || this.momentum > 0.0) && (this.g_sum.length == 0)) {
					for (var i=0; i<pg_ls.length; i++) {
						this.g_sum.push(global.zeros(pg_ls[i].params.length));
						if (this.optimizer == 'adadelta') {
							this.x_sum.push(global.zeros(pg_ls[i].params.length));
						} else {
							this.x_sum.push([]);
						}
					}
				}

				for (var i=0; i<pg_ls.length; i++) {
					var pg = pg_ls[i];
					var p = pg.params;
					var g = pg.grads;

					var l1_decay_mul = typeof pg.l1_decay_mul == 'undefined' ? 1.0 : pg.l1_decay_mul;
					var l2_decay_mul = typeof pg.l2_decay_mul == 'undefined' ? 1.0 : p2.l2_decay_mul;

					var l1_decay = this.l1_decay * l1_decay_mul;
					var l2_decay = this.l2_decay * l2_decay_mul;

					for (var j=0; j<p.length; j++) {
						l1_decay_loss += l1_decay * Math.abs(p[j]);
						l2_decay_loss += l2_decay * p[j] * p[j] / 2;
						var l1_grad = l1_decay * (p[j] > 0 ? 1.0 : -1.0 );
						var l2_grad = l2_decay * p[j];

						var g_i_j = (g[j] + l1_grad + l2_grad) / this.batch_size;

						var g_sum_i = this.g_sum[i];
						var x_sum_i = this.x_sum[i];

						if (this.optimizer == 'adagrad') {
							// Adagrad
							var sub_g_i_j = g_i_j * g_i_j + g_sum_i[j];
							g_sum_i[j] = sub_g_i_j;
							p[j] -= this.learning_rate / Math.sqrt(sub_g_i_j + this.epsilon) * g_i_j;
						} else if (this.optimizer == 'adadelta') {
							// Adadelta
							g_sum_i[j] = this.ro * g_sum_i[j] + (1 - this.ro) * g_i_j * g_i_j;
							var x_grad_i_j = - Math.sqrt(x_sum_i[j] + this.epsilon) / Math.sqrt(g_sum_i[j] + this.epsilon) * g_i_j;

							x_sum_i[j] = this.ro * x_sum_i[j] + (1 - this.ro) * x_grad_i_j * x_grad_i_j;
							p[j] += x_grad_i_j;
						} else {
							// Stochastic Gradient Descent
							// this is an amazing paper on momentum: https://distill.pub/2017/momentum/
							if (this.momentum > 0.0) {
								var z = this.momentum * g_sum_i[j] + g_i_j * this.learning_rate;
								g_sum_i[j] = z;
								p[j] -= z;
							} else {
								p[j] -= this.learning_rate * g_i_j;
							}
						}
						// always 0 out the gradient
						g[j] = 0.0;
					}
				}
			}
			return {
				forward_time: forward_time,
				backward_time: backward_time,
				l1_decay_loss: l1_decay_loss,
				l2_decay_loss: l2_decay_loss,
				cost_loss: cost_loss,
				loss: cost_loss + l1_decay_loss + l2_decay_loss
			}
		}
	}

	global.Tensor = Tensor;
	global.InputLayer = InputLayer;
	global.DenseLayer = DenseLayer;
	global.LinearLayer = LinearLayer;
	global.SigmoidLayer = SigmoidLayer;
	global.TanhLayer = TanhLayer;
	global.ReluLayer = ReluLayer;
	global.DropoutLayer = DropoutLayer;
	global.SoftmaxLayer = SoftmaxLayer;
	global.Model = Model;
	global.Trainer = Trainer;

	global.randr = randr;
	global.randi = randi;
	global.randGauss = randGauss;
	global.randn = randn;
	global.tanh = tanh;



})(browsernn);

(function(lib) {
	"use strict";
	if (typeof module == "undefined" || typeof module.exports == "undefined") {
		window.jsfeat = lib;
	} else {
		module.exports = lib;
	}
})(browsernn);
