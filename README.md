# An Efficient Implementation of Kolmogorov-Arnold Network using Chebyshev polynomials

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN) forked from [efficent-kan](https://github.com/Blealtan/efficient-kan).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan), based on [this](https://arxiv.org/abs/2404.19756) paper.

The difference between this repo and [efficent-kan](https://github.com/Blealtan/efficient-kan) is that this version uses Chebyshev polynomials rather than splines for the base function fitting. Chebyshev polynomials are a classic approach in function approximation, and are much easier to interpret in terms of their functional form than splines. Chebyshev polynomials also do not require a grid to be specified, reducing the number of assumptions on data structure.

On toy systems Chebyshev polynomials are comparable in performance to splines. 

Several further steps of improvement would be required before this version could be production-ready:
- Use more efficient (but less interpretable) Clenshaw algorithm for evaluation of Chebyshev sums
- Understand requirements on initialization of weight variables
- Improve regularization procedure, probably along the lines of penalizing higher-order Chebyshev terms as suggested here
- Enable range of input data to be specified rather than necessarily [-1,1]
- Other base function choices are possible: for example [Hermite functions](https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions) are utilized on [-inf,inf] rather than [-1,1] and the [Legendre Polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials) are orthogonal over [-1,1] with respect to a simpler weight measure than Chebyshev polynomials
