(function(global) {
  "use strict";

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

  global.Trainer = Trainer;

})(browsernn);
