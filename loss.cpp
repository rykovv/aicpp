#include <concepts>
#include <iostream>

#include <cmath>
#include <ranges>
#include <vector>

namespace loss {
    template <typename Range>
    static constexpr typename Range::value_type L1 (Range const& ground, Range const& predicted) {
        typename Range::value_type l1 = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            l1 += std::abs(gnd - pred);
        }
        return l1;
    }

    template <typename Range>
    static constexpr typename Range::value_type L2 (Range const& ground, Range const& predicted) {
        typename Range::value_type l2 = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            l2 += std::pow(gnd - pred, 2);
        }
        return l2;
    }

    template <typename Range>
    static constexpr typename Range::value_type huber (Range const& ground, Range const& predicted, typename Range::value_type const threshold) {
        using value_type_t = typename Range::value_type;
        value_type_t huber = 0;
        auto hbr = [&threshold] (value_type_t diff) {
            if (diff <= threshold) {
                return std::pow(diff, 2)/2;
            } else {
                return threshold*std::abs(diff) - threshold/2;
            }
        };
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            huber += hbr(gnd - pred);
        }
        return huber;
    }

    template <typename Range>
    static constexpr typename Range::value_type bce (Range const& ground, Range const& predicted) {
        // binary_cross_entropy
        using value_type_t = typename Range::value_type;
        value_type_t bce = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            bce += gnd*std::log(pred) + (gnd - 1)*log(1 - pred);
        }
        return -bce/std::ranges::size(ground);
    }
}

int main () {
    std::vector<double> ground = {0.1, 1.0, 0.3, 0.5, 0.7};
    std::vector<double> predicted = {0.1, 0.3, 0.4, 0.1, 0.2};
    std::cout << "L1 = " << loss::L1(ground, predicted) << std::endl;
    std::cout << "L2 = " << loss::L2(ground, predicted) << std::endl;
    std::cout << "Huber = " << loss::huber(ground, predicted, 0.2) << std::endl;
    std::cout << "BCE = " << loss::bce(ground, predicted) << std::endl;
    return 0;
}
