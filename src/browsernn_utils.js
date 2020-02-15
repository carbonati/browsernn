(function(global) {
	"use strict";

	function zeros(n) {
		if(typeof ArrayBuffer === 'undefined') {
	      // lacking browser support
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
	function flatten(arr) {
		return arr.reduce(function (flat, toFlatten) {
			return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
		}, []);
	}
	function tanh(x) {
		return (Math.exp(2*x) - 1) / (Math.exp(2*x) + 1);
	}

	global.zeros = zeros;
	global.random = random;
	global.randr = randr;
	global.randi = randi;
	global.randGauss = randGauss;
	global.randn = randn;
	global.flatten = flatten;
	global.tanh = tanh;

})(browsernn);
