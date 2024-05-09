#include <concepts>
#include <iostream>

#include <cmath>
#include <ranges>
#include <algorithm>
#include <vector>

namespace loss {

    template <typename Range, typename T = typename Range::value_type, std::invocable<T, T> F>
    static constexpr T _apply_and_sum (F f, Range const& r0, Range const& r1) {
        auto applied = std::views::zip_transform(f, r0, r1);
        return std::ranges::fold_right(applied.begin(), applied.end(), 0, std::plus<>());
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T L1 (Range const& ground, Range const& predicted) {
        T l1 = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            l1 += std::abs(gnd - pred);
        }
        return l1;
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T L1_f (Range const& ground, Range const& predicted) {
        auto abs_diff = [](T a, T b) -> T { 
            return std::abs(a - b); 
        };
        return loss::_apply_and_sum(abs_diff, ground, predicted);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T L2 (Range const& ground, Range const& predicted) {
        T l2 = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            l2 += std::pow(gnd - pred, 2);
        }
        return std::sqrt(l2);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T L2_f (Range const& ground, Range const& predicted) {
        auto euc_dist = [](T a, T b) -> T { 
            return std::pow(a - b, 2);
        };
        return std::sqrt(loss::_apply_and_sum(euc_dist, ground, predicted));
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T huber (Range const& ground, Range const& predicted, T const threshold) {
        T huber = 0;

        auto hbr = [&threshold](T diff) -> T {
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

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T huber_f (Range const& ground, Range const& predicted, T const threshold) {
        auto hbr = [&threshold](T a, T b) -> T {
            T diff = a - b;
            if (diff <= threshold) {
                return std::pow(diff, 2)/2;
            } else {
                return threshold*std::abs(diff) - threshold/2;
            }
        };
        return loss::_apply_and_sum(hbr, ground, predicted);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T bce (Range const& ground, Range const& predicted) {
        // binary_cross_entropy
        T bce = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            bce += gnd*std::log(pred) + (gnd - 1)*log(1 - pred);
        }
        return -bce/std::ranges::size(ground);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T bce_f (Range const& ground, Range const& predicted) {
        // binary_cross_entropy
        auto f = [](T gnd, T pred) -> T {
            return gnd*std::log(pred) + (gnd - 1)*log(1 - pred);
        };
        T bce = loss::_apply_and_sum(f, ground, predicted);
        return -bce/std::ranges::size(ground);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T ce (Range const& ground, Range const& predicted) {
        // cross_entropy
        T ce = 0;
        for (auto&& [gnd, pred] : std::ranges::views::zip(ground, predicted)){
            ce += gnd*std::log(pred);
        }
        return -ce/std::ranges::size(ground);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T ce_f (Range const& ground, Range const& predicted) {
        // cross_entropy
        auto f = [](T gnd, T pred) -> T {
            return gnd*std::log(pred);
        };

        
        T ce = loss::_apply_and_sum(f, ground, predicted);
        return -ce/std::ranges::size(ground);
    }

    template <typename Range>
    static constexpr Range softmax (Range const& predicted) {
        using value_type_t = typename Range::value_type;
        
        // exponent all values
        auto expd = predicted | std::views::transform(exp);
        // sum of expd values
        auto exp_sum = std::ranges::fold_right(expd.begin(), expd.end(), 0, std::plus<>());
        
        auto div_by_sum = [&exp_sum](auto pred) -> value_type_t {
            return pred/exp_sum;
        };

        return expd | std::views::transform(div_by_sum) | std::ranges::to<Range>();
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T kl (Range const& ground, Range const& predicted) {
        // KL divergence
        auto f = [](T gnd, T pred) -> T {
            return gnd*std::log(gnd/pred);
        };

        return loss::_apply_and_sum(f, ground, predicted);
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T contrastive (bool ground, Range const& featuresA, Range const& featuresB, T const margin) {
        T dist = loss::L2_f(featuresA, featuresB);
        using std::max, std::pow;

        return [&]() -> T {
            if (ground) {
                return pow(dist, 2);
            } else{
                T zero{0};
                return pow(max(margin - dist, zero), 2);
            }
        }();
    }

    template <typename Range, typename T = typename Range::value_type>
    static constexpr T hinge (Range const& ground, Range const& predicted) {
        auto f = [](T gnd, T pred) -> T {
            return std::max(0, 1 - gnd*pred);
        };

        return loss::_apply_and_sum(f, ground, predicted);
    }
}

int main () {
    std::vector<double> ground = {0.1, 1.0, 0.3, 0.5, 0.7};
    std::vector<double> predicted = {0.1, 0.3, 0.4, 0.1, 0.2};

    auto print_range = []<typename Range>(Range const& r) -> void {
        std::cout << "{ ";
        for (auto v : r) {
            std::cout << v << " ";
        }
        std::cout << "}" << std::endl;
    };

    std::cout << "L1 = " << loss::L1(ground, predicted) << std::endl;
    std::cout << "L1_f = " << loss::L1_f(ground, predicted) << std::endl;
    std::cout << "L2 = " << loss::L2(ground, predicted) << std::endl;
    std::cout << "L2_f = " << loss::L2_f(ground, predicted) << std::endl;
    std::cout << "Huber = " << loss::huber(ground, predicted, 0.2) << std::endl;
    std::cout << "Huber_f = " << loss::huber_f(ground, predicted, 0.2) << std::endl;
    std::cout << "BCE = " << loss::bce(ground, predicted) << std::endl;
    std::cout << "BCE_f = " << loss::bce_f(ground, predicted) << std::endl;
    std::cout << "CE = " << loss::ce(ground, predicted) << std::endl;
    std::cout << "CE_f = " << loss::ce_f(ground, predicted) << std::endl;
    std::cout << "softmax = "; print_range(loss::softmax(predicted));
    std::cout << "KL = " << loss::kl(ground, predicted) << std::endl;
    std::cout << "contrastive = " << loss::contrastive(1, ground, predicted, 2.0) << std::endl;
    
    return 0;
}
