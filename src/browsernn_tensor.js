(function(global) {
  "use strict";

  var Tensor = function(n, d, depth, init_weight, seed) {
    // args:
      // n: number of rows
      // d: number of columns
      // depth: depth of the tensor
        // if only 2 dimensions are passed as (n,d) the Tensor will
                // create a depth of 1 as Tensor(1, 2, 1)
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

    this.w = global.zeros(this.n_cells);
    this.dw = global.zeros(this.n_cells);

    // weight normalized parameterization - to reduce variance over the
        // distribution of weights
    if(typeof init_weight == 'undefined') {
      var std = Math.sqrt(1.0/(this.n_cells));
      for (var i=0; i<this.n_cells; i++) {
        // hack for initializing weights that differ between layers
        var reseed = seed+this.n_cells+i;
                if (typeof seed != 'undefined') {
                    self.w[i] = global.randn(0.0, std, reseed)
                } else {
                   global.randn(0.0, std, undefined)
                }
      }
    } else if (Object.prototype.toString.call(init_weight) == '[object Array]') {
      var weights_flat = global.flatten(init_weight);
      for (var i=0; i<n_cells; i++) {
        this.w[i] = weights_flat[i];
      }
    } else {
      for (var i=0; i<n_cells; i++) {
        this.w[i] = init_weight;
      }
    }

  }

  global.Tensor = Tensor;

})(browsernn);
