#include <iostream>
#include <concepts>
#include <cmath>

namespace activation {
    template <std::floating_point T>
    static constexpr T sigmoid (T const& z) {
        // bad for activation due to vanishing gradient
        // okay for gating functions
        using std::exp;
        return T{1}/(T{1}+exp(-z));
    }

    template <std::floating_point T>
    static constexpr T tanh (T const& z) {
        // zero-centered (better than sigmoid)
        // used in recurrent nn and lstm
        using std::tanh;
        return tanh(z);
    }

    template <std::floating_point T>
    static constexpr T relu (T const& z) {
        // most popular, best performance in cnn
        using std::max;
        return max(T{0}, z);
    }

    template <std::floating_point T>
    static constexpr T prelu (T const& z, T const& alpha) {
        // parameteric relu
        return z > T{0} ? z : z*alpha;
    }

    template <std::floating_point T>
    static constexpr T elu (T const& z, T const& alpha) {
        // exponentially linear unit
        using std::exp;
        return z > T{0} ? z : alpha*(exp(z) - 1);
    }

    template <std::floating_point T>
    static constexpr T glu (T const& z) {
        // gated linear unit
        return z*activation::sigmoid(z);
    }

    template <std::floating_point T>
    static constexpr T swish (T const& z) {
        // sparsity, no saturation
        // small negativesa are not zero'd out
        return activation::glu(z);
    }

    template <std::floating_point T>
    static constexpr T softplus (T const& z, T const& beta) {
        using std::log, std::exp;
        return log(T{1} + exp(z*beta))/beta;
    }

    template <std::floating_point T>
    static constexpr T mish (T const& z) {
        // no saturation, continuous
        // small negativesa are not zero'd out
        using std::tanh;
        return z*tanh(activation::softplus(z, T{1}));
    }
}

int main () {
    std::cout << "sigmoid(2) = " << activation::sigmoid(2.0) << std::endl;
    std::cout << "tanh(2) = " << activation::tanh(2.0) << std::endl;
    std::cout << "relu(2) = " << activation::relu(2.0) << std::endl;
    std::cout << "prelu(2) = " << activation::prelu(2.0, 0.1) << std::endl;
    std::cout << "elu(2) = " << activation::elu(2.0, 0.1) << std::endl;
    std::cout << "glu(2) = " << activation::glu(2.0) << std::endl;
    std::cout << "swish(2) = " << activation::swish(2.0) << std::endl;
    std::cout << "softplus(2) = " << activation::softplus(2.0, 0.1) << std::endl;
    std::cout << "mish(2) = " << activation::mish(2.0) << std::endl;

    return 0;
}
