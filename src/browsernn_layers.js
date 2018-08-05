(function(global) {
	"use strict";

	var Tensor = global.Tensor;

	var InputLayer = function(params) {
		// args;
			// params: object containing the dimensions for the input layer of the model
				// in the case of 2 dimension data like (x_1, x_2) the Tensor should take the shape Tensor(1, 2, 1)

		// returns:
			// instantiates the models input layer
		
		var params = params || {}; // if nothing is passed set param to be an empty object

		if((params.depth < 1) || (typeof params.depth == 'undefined')) params.depth = 1;

		// i technically don't need getparams()
		this.out_n = params.n;
		this.out_d = params.d;
		this.out_depth = params.depth;

		this.layer_type = 'input';
	}

	InputLayer.prototype = {
		forward: function(tensor, trainable) {
			// args:
				// tensor: tensor holding the input data (X)
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// output of the input layer

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
			// params: hyper parameters of the dense layer that must include the number of neurons and activation function
			// seed: seed to reinitialize random variables

		// returns:
			// instantiated dense layer
		
		var params = params || {}; // if nothing is passed set param to be an empty object
		this.n_neurons = params.n_neurons;

		this.l1_decay_mul = typeof params.l1_decay_mul != 'undefined' ? 0.0 : params.l1_decay_mul;
		this.l2_decay_mul = typeof params.l2_decay_mul != 'undefined' ? 1.0 : params.l2_decay_mul;

		this.n_inputs = params.in_n * params.in_d * params.in_depth;
		this.out_n = 1;
		this.out_d = params.n_neurons;
		this.out_depth = 1;
		this.layer_type = 'dense';

		var bias = typeof params.bias_pref == 'undefined' ? 0.0 : params.bias_pref;

		this.weight_mat = []; // this arrary will hold all the weights connected from each node between layer[i-1] and layer[i]
		
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
				// linear transformation between neurons from the previous layer and the weights form the dense layer

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
				var chain_grad = this.out_units.dw[i]; // gradient from previous layer
				for (var j=0; j<this.n_inputs; j++) {
					input.dw[j] += chain_grad * weight_vec.w[j]; // gradient wrt inputs
					weight_vec.dw[j] += chain_grad * input.w[j]; // gradient wrt weights
				}
			
			this.biases.dw[i] += chain_grad;
			}			
		},

		getParamsAndGrads: function() {
			// args:

			// returns:
				// list containing the layers weights, derivatives, and regularized multipliers

			var pg_ls = [];
			for (var i=0; i<this.n_neurons; i++) {
				pg_ls.push({ 
					params: this.weight_mat[i].w, 
					grads: this.weight_mat[i].dw,
					l1_decay_mul: this.l1_decay_mul,
					l2_decay_mul: this.l2_decay_mul
				});
			}

			pg_ls.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
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
				// tensor: tensor holding data from the previous layer (z)
				// trainable: (boolean) whether the model is training or predicting

			// returns:
				// output tensor from the previous layer - since the dense computes a linear transformation

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

		// when this function is called we will create a DenseLayer as well to connect the previous layer, 
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

			// the softmax becomes saturated when the inputs are extremely positive or negative
			// also becomes saturated when the difference beween inputs is extremely large
			// invariant to adding some scalar value (i.e. the maximum input)
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

	global.InputLayer = InputLayer;
	global.DenseLayer = DenseLayer;
	global.LinearLayer = LinearLayer;
	global.SigmoidLayer = SigmoidLayer;
	global.TanhLayer = TanhLayer;
	global.ReluLayer = ReluLayer;
	global.SoftmaxLayer = SoftmaxLayer;

})(browsernn);