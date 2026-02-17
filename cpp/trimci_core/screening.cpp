#include "screening.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <atomic>
#include <iostream>
#include <fstream>
#include <chrono>
#ifdef _OPENMP
#  include <omp.h>
#endif
#include "omp_compat.hpp"
#include "bit_compat.hpp"

namespace trimci_core {

// Helper Functions - Precompute double excitation table
// If attentive_orbitals is non-empty, only include (i,j,p,q) where all are in the set
// OPTIMIZED: Parallelized using OpenMP for large systems
DoubleExcTable precompute_double_exc_table(
    int n_orb,
    const std::vector<double>& eri,
    double thr,
    const std::vector<int>& attentive_orbitals
) {
    // Build attentive set for O(1) lookup
    std::unordered_set<int> att_set(attentive_orbitals.begin(), attentive_orbitals.end());
    bool use_attentive = !att_set.empty();
    
    // Determine which orbitals to iterate over
    std::vector<int> orb_list;
    if (use_attentive) {
        orb_list = attentive_orbitals;
        std::sort(orb_list.begin(), orb_list.end());
    } else {
        orb_list.reserve(n_orb);
        for (int o = 0; o < n_orb; ++o) orb_list.push_back(o);
    }
    
    size_t n = orb_list.size();
    
    // Collect all (i,j) pairs for parallel processing
    std::vector<std::pair<int,int>> ij_pairs;
    ij_pairs.reserve(n * (n-1) / 2);
    for (size_t ai = 0; ai < n; ++ai) {
        for (size_t aj = ai + 1; aj < n; ++aj) {
            ij_pairs.push_back({orb_list[ai], orb_list[aj]});
        }
    }
    
    // Parallel compute entries for each (i,j) pair
    std::vector<std::vector<std::tuple<int,int,double>>> results(ij_pairs.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < (int64_t)ij_pairs.size(); ++idx) {
        int i = ij_pairs[idx].first;
        int j = ij_pairs[idx].second;
        std::vector<std::tuple<int,int,double>> entries;
        
        for (size_t ap = 0; ap < n; ++ap) {
            int p = orb_list[ap];
            for (size_t aq = ap + 1; aq < n; ++aq) {
                int q = orb_list[aq];
                double h_val = get_eri(eri, n_orb, i, p, j, q) - get_eri(eri, n_orb, i, q, j, p);
                if (std::abs(h_val) > thr) {
                    entries.emplace_back(p, q, h_val);
                }
            }
        }
        
        // Sort by |h_val| descending
        std::sort(entries.begin(), entries.end(),
                  [](auto& a, auto& b){
                      return std::abs(std::get<2>(a)) > std::abs(std::get<2>(b));
                  });
        results[idx] = std::move(entries);
    }
    
    // Merge results into table
    DoubleExcTable table;
    for (size_t idx = 0; idx < ij_pairs.size(); ++idx) {
        if (!results[idx].empty()) {
            table[ij_pairs[idx]] = std::move(results[idx]);
        }
    }
    return table;
}

// Template implementation for processing parent determinants
// If attentive_set is non-empty, all excitations are restricted to those orbitals
template<typename StorageType>
std::vector<std::pair<DeterminantT<StorageType>, double>>
process_parent_worker_t(
    const DeterminantT<StorageType>& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::unordered_set<int>& attentive_set,
    const IntegralSparsityInfo* sparsity
) {
    std::vector<std::pair<DeterminantT<StorageType>, double>> new_pairs;
    auto occ_a = det.getOccupiedAlpha();
    auto occ_b = det.getOccupiedBeta();
    
    bool use_attentive = !attentive_set.empty();

    // αα same-spin double excitations
    for (size_t ia = 0; ia < occ_a.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_a.size(); ++ib) {
            int i = occ_a[ia], j = occ_a[ib];
            // Skip if i or j not in attentive set
            if (use_attentive && (attentive_set.find(i) == attentive_set.end() || 
                                  attentive_set.find(j) == attentive_set.end())) continue;
            auto it = table.find({i,j});
            if (it == table.end()) continue;
            
            for (const auto& t : it->second) {
                int p, q;
                double h_val;
                std::tie(p, q, h_val) = t;
                
                if (BitOps<StorageType>::get_bit(det.alpha, p) || 
                    BitOps<StorageType>::get_bit(det.alpha, q)) continue;
                
                auto dj = det.doubleExcite(i, j, p, q, true);
                int ph = detail::double_phase_t(det.alpha, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }
    
    // ββ same-spin double excitations
    for (size_t ia = 0; ia < occ_b.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_b.size(); ++ib) {
            int i = occ_b[ia], j = occ_b[ib];
            // Skip if i or j not in attentive set
            if (use_attentive && (attentive_set.find(i) == attentive_set.end() || 
                                  attentive_set.find(j) == attentive_set.end())) continue;
            auto it = table.find({i,j});
            if (it == table.end()) continue;
            
            for (const auto& t : it->second) {
                int p, q;
                double h_val;
                std::tie(p, q, h_val) = t;
                
                if (BitOps<StorageType>::get_bit(det.beta, p) || 
                    BitOps<StorageType>::get_bit(det.beta, q)) continue;
                
                auto dj = det.doubleExcite(i, j, p, q, false);
                int ph = detail::double_phase_t(det.beta, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }
    
    // Mixed αβ double excitations
    if (sparsity && sparsity->is_sparse) {
        // Sparse path: use precomputed ab_exc_table
        for (int i : occ_a) {
            if (use_attentive && attentive_set.find(i) == attentive_set.end()) continue;
            for (const auto& entry : sparsity->ab_exc_table[i]) {
                int j = std::get<0>(entry);
                int p = std::get<1>(entry);
                int q = std::get<2>(entry);
                double eri_val = std::get<3>(entry);
                // j must be occupied in beta
                if (!BitOps<StorageType>::get_bit(det.beta, j)) continue;
                // p must be virtual in alpha
                if (BitOps<StorageType>::get_bit(det.alpha, p)) continue;
                // q must be virtual in beta
                if (BitOps<StorageType>::get_bit(det.beta, q)) continue;
                // Attentive checks
                if (use_attentive) {
                    if (attentive_set.find(j) == attentive_set.end()) continue;
                    if (attentive_set.find(p) == attentive_set.end()) continue;
                    if (attentive_set.find(q) == attentive_set.end()) continue;
                }
                if (std::abs(eri_val) <= thr) continue;

                StorageType new_alpha = det.alpha;
                StorageType new_beta = det.beta;
                BitOps<StorageType>::clear_bit(new_alpha, i);
                BitOps<StorageType>::set_bit(new_alpha, p);
                BitOps<StorageType>::clear_bit(new_beta, j);
                BitOps<StorageType>::set_bit(new_beta, q);

                DeterminantT<StorageType> dj(new_alpha, new_beta);
                int pa = detail::single_phase_t(det.alpha, i, p);
                int pb = detail::single_phase_t(det.beta, j, q);
                new_pairs.emplace_back(dj, pa * pb * eri_val);
            }
        }
    } else {
        // Dense path (original)
        for (int i : occ_a) {
            if (use_attentive && attentive_set.find(i) == attentive_set.end()) continue;
            for (int j : occ_b) {
                if (use_attentive && attentive_set.find(j) == attentive_set.end()) continue;
                auto iterate_p = [&](auto&& callback) {
                    if (use_attentive) {
                        for (int p : attentive_set) callback(p);
                    } else {
                        for (int p = 0; p < n_orb; ++p) callback(p);
                    }
                };
                iterate_p([&](int p) {
                    if (BitOps<StorageType>::get_bit(det.alpha, p)) return;
                    auto iterate_q = [&](auto&& callback_q) {
                        if (use_attentive) {
                            for (int q : attentive_set) callback_q(q);
                        } else {
                            for (int q = 0; q < n_orb; ++q) callback_q(q);
                        }
                    };
                    iterate_q([&](int q) {
                        if (BitOps<StorageType>::get_bit(det.beta, q)) return;
                        double h_val = get_eri(eri, n_orb, i, p, j, q);
                        if (std::abs(h_val) <= thr) return;
                        StorageType new_alpha = det.alpha;
                        StorageType new_beta = det.beta;
                        BitOps<StorageType>::clear_bit(new_alpha, i);
                        BitOps<StorageType>::set_bit(new_alpha, p);
                        BitOps<StorageType>::clear_bit(new_beta, j);
                        BitOps<StorageType>::set_bit(new_beta, q);
                        DeterminantT<StorageType> dj(new_alpha, new_beta);
                        int pa = detail::single_phase_t(det.alpha, i, p);
                        int pb = detail::single_phase_t(det.beta, j, q);
                        new_pairs.emplace_back(dj, pa * pb * h_val);
                    });
                });
            }
        }
    }
    
    // α single excitations
    for (int i : occ_a) {
        if (use_attentive && attentive_set.find(i) == attentive_set.end()) continue;

        if (sparsity && sparsity->eri_is_diagonal) {
            // Sparse path: only h1 neighbors (ERI terms vanish for singles)
            for (int p : sparsity->h1_neighbors[i]) {
                if (use_attentive && attentive_set.find(p) == attentive_set.end()) continue;
                if (BitOps<StorageType>::get_bit(det.alpha, p)) continue;
                auto dj = det.singleExcite(i, p, true);
                double hij = compute_H_ij_t(det, dj, h1, eri, sparsity);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            }
        } else {
            // Dense path (original)
            auto iterate_p = [&](auto&& callback) {
                if (use_attentive) {
                    for (int p : attentive_set) callback(p);
                } else {
                    for (int p = 0; p < n_orb; ++p) callback(p);
                }
            };
            iterate_p([&](int p) {
                if (BitOps<StorageType>::get_bit(det.alpha, p)) return;
                auto dj = det.singleExcite(i, p, true);
                double hij = compute_H_ij_t(det, dj, h1, eri);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            });
        }
    }

    // β single excitations
    for (int j : occ_b) {
        if (use_attentive && attentive_set.find(j) == attentive_set.end()) continue;

        if (sparsity && sparsity->eri_is_diagonal) {
            // Sparse path: only h1 neighbors
            for (int q : sparsity->h1_neighbors[j]) {
                if (use_attentive && attentive_set.find(q) == attentive_set.end()) continue;
                if (BitOps<StorageType>::get_bit(det.beta, q)) continue;
                auto dj = det.singleExcite(j, q, false);
                double hij = compute_H_ij_t(det, dj, h1, eri, sparsity);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            }
        } else {
            // Dense path (original)
            auto iterate_q = [&](auto&& callback) {
                if (use_attentive) {
                    for (int q : attentive_set) callback(q);
                } else {
                    for (int q = 0; q < n_orb; ++q) callback(q);
                }
            };
            iterate_q([&](int q) {
                if (BitOps<StorageType>::get_bit(det.beta, q)) return;
                auto dj = det.singleExcite(j, q, false);
                double hij = compute_H_ij_t(det, dj, h1, eri);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            });
        }
    }

    return new_pairs;
}

// Wrapper for backward compatibility (process_parent_worker)
std::vector<std::pair<Determinant,double>>
process_parent_worker(
    const Determinant& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri
) {
    // Pass empty attentive_set for backward compatibility (use all orbitals)
    return process_parent_worker_t<uint64_t>(det, n_orb, thr, table, h1, eri, {});
}


// Template implementation for pool building
// Two-stage screening for PT2 modes: heat_bath pre-filter -> PT2 refinement
template<typename StorageType>
std::pair<std::vector<DeterminantT<StorageType>>, double>
pool_build_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    const std::vector<double>& initial_coeffs,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCacheT<StorageType>& cache,
    const std::string& cache_file,
    const std::vector<int>& attentive_orbitals,
    int verbosity,
    const PoolBuildParams& params
) {
    // Extract parameters from struct
    const std::string& screening_mode = params.screening_mode;
    double e0 = params.e0;
    int max_rounds = params.max_rounds;
    double threshold_decay = params.threshold_decay;
    double min_threshold = params.min_threshold;
    int max_stagnant_rounds_param = params.max_stagnant_rounds;
    double pt2_denom_min = params.pt2_denom_min;
    
    // Determine effective strategy factor
    int effective_factor;
    if (params.strategy_factor > 0) {
        effective_factor = params.strategy_factor;
    } else {
        // Auto: 1 for heat_bath, 20 for PT2 modes
        if (screening_mode == "heat_bath") {
            effective_factor = 1;
        } else {
            effective_factor = 20;
        }
    }
    size_t effective_target = target_size * effective_factor;
    
    // Determine screening mode flags
    bool use_pt2_denominator = (screening_mode == "heat_bath_pt2" || screening_mode == "pt2");
    bool use_pt2_aggregation = (screening_mode == "pt2");
    
    if (verbosity >= 1) {
        std::cout << "[PoolBuild] Starting pool build: "
              << "target_size=" << target_size
              << ", threshold=" << threshold
              << ", mode=" << screening_mode
              << ", factor=" << effective_factor;
        if (use_pt2_denominator) {
            std::cout << ", e0=" << e0;
        }
        if (!attentive_orbitals.empty()) {
            std::cout << ", attentive_orbitals=" << attentive_orbitals.size();
        }
        std::cout << std::endl;
    }

    // Build attentive set for O(1) lookup
    std::unordered_set<int> attentive_set(attentive_orbitals.begin(), attentive_orbitals.end());

    auto precompute_start = std::chrono::high_resolution_clock::now();
    auto table = precompute_double_exc_table(n_orb, eri, threshold, attentive_orbitals);
    auto precompute_end = std::chrono::high_resolution_clock::now();
    double precompute_time = std::chrono::duration<double>(precompute_end - precompute_start).count();
    if (verbosity >= 2) {
        std::cout << "[PoolBuild] precompute_double_exc_table: " << precompute_time << "s, entries=" << table.size() << std::endl;
    }

    // Build integral sparsity info (one-time cost)
    auto sparsity_info = build_sparsity_info(n_orb, h1, eri);
    const IntegralSparsityInfo* sparsity_ptr = sparsity_info.is_sparse ? &sparsity_info : nullptr;
    if (verbosity >= 1 && sparsity_info.is_sparse) {
        std::cout << "[PoolBuild] Sparsity detected: h1=" << sparsity_info.h1_sparsity
                  << ", eri=" << sparsity_info.eri_sparsity
                  << ", diagonal=" << sparsity_info.eri_is_diagonal << std::endl;
    }

    std::unordered_set<DeterminantT<StorageType>> pool_set(
        initial_pool.begin(), initial_pool.end()
    );
    pool_set.reserve(target_size);
    std::vector<DeterminantT<StorageType>> frontier = initial_pool;
    
    // Control whether to use coefficient map
    bool use_coeffs = !initial_coeffs.empty();
    if (verbosity >= 2) {
        std::cout << "[PoolBuild] use_coeffs: " << use_coeffs << ", screening_mode: " << screening_mode << std::endl;
    }

    std::unordered_map<DeterminantT<StorageType>, double> coeff_map;
    if (use_coeffs) {
        coeff_map.reserve(initial_pool.size());
        for (size_t i = 0; i < initial_pool.size(); ++i) {
            coeff_map[initial_pool[i]] = initial_coeffs[i];
        }
    }

    int round = 1;
    std::atomic<bool> reached{false};
    size_t prev_pool_size = pool_set.size();
    int stagnant_rounds = 0;

    while (pool_set.size() < target_size) {
        if (frontier.empty() || (max_rounds > 0 && round > max_rounds)) {
            if (pool_set.size() < target_size) {
                // Check if threshold is too small or no progress is being made
                if (threshold < min_threshold) {
                if (verbosity >= 1) {
                    std::cout << "[PoolBuild] threshold too small (" << threshold 
                              << " < " << min_threshold << "), stopping to prevent infinite loop." << std::endl;
                }
                    break;
                }
                if (pool_set.size() == prev_pool_size) {
                    stagnant_rounds++;
                    if (stagnant_rounds >= max_stagnant_rounds_param) {
                        if (verbosity >= 1) {
                            std::cout << "[PoolBuild] no progress for " << max_stagnant_rounds_param 
                                  << " threshold reductions, stopping." << std::endl;
                        }
                        break;
                    }
                } else {
                    stagnant_rounds = 0;
                    prev_pool_size = pool_set.size();
                }
                
                // Relax threshold and restart
                threshold *= threshold_decay;
                // recompute double excitation table with new threshold
                table = precompute_double_exc_table(n_orb, eri, threshold, attentive_orbitals);
                round = 1;
                frontier = std::vector<DeterminantT<StorageType>>(initial_pool.begin(), initial_pool.end());
                if (verbosity >= 1) {
                    std::cout << "[PoolBuild] threshold relaxed to " << threshold
                          << ", restarting from initial pool=" << initial_pool.size()
                          << std::endl;
                }
            } else {
                break;
            }
        }

        if (verbosity >= 2) {
            std::cout << "[PoolBuild] Round " << round
                  << ": pool_size=" << pool_set.size()
                  << ", frontier_size=" << frontier.size()
                  << std::endl;
        }

        std::vector<DeterminantT<StorageType>> new_frontier;
        
        // Candidate definition for deferred processing
        // Added h_jj for PT2 denominator, screening_score for final ranking
        struct Candidate {
            DeterminantT<StorageType> parent;
            DeterminantT<StorageType> child;
            double hij;
            double est_cj;
            double h_jj;            // Diagonal element H_jj (for PT2)
            double screening_score; // Final score used for selection
        };
        
        #ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        #else
        int n_threads = 1;
        #endif
        
        // Vector of vectors to buffer candidates and avoid lock contention (Deferred Merge Strategy)
        std::vector<std::vector<Candidate>> thread_candidates(n_threads);

        // Stage 1: Heat-bath pre-collection (unified for all strategies)
        // All strategies use the same threshold-based collection
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_cands = thread_candidates[tid];

            #pragma omp for schedule(dynamic)
            for (int idx = 0; idx < (int)frontier.size(); ++idx) {
                auto det = frontier[idx];
                double ci = 1.0;
                if (use_coeffs) {
                    auto it = coeff_map.find(det);
                    ci = (it != coeff_map.end()) ? it->second : 1.0;
                }
                double abs_ci = std::abs(ci);
                double local_threshold = threshold / std::max(abs_ci, 1e-12);

                auto locals = process_parent_worker_t(det, n_orb, local_threshold, table, h1, eri, attentive_set, sparsity_ptr);

                for (auto& pr : locals) {
                    const auto& dj = pr.first;
                    double hij = pr.second;
                    
                    // Stage 1: Use heat_bath score for pre-filtering
                    if (std::abs(hij) > local_threshold) {
                        double heat_bath_score = std::abs(hij * abs_ci);
                        // h_jj = 0 for now, will compute in Stage 2 for PT2 modes
                        local_cands.push_back({det, dj, hij, abs_ci, 0.0, heat_bath_score});
                    }
                }
            }
        } // omp parallel end

        // Stage 2: Merge and refine
        if (use_pt2_denominator) {
            // PT2 modes: Two-stage screening
            // First collect all candidates
            std::vector<Candidate> all_candidates;
            for (int t = 0; t < n_threads; ++t) {
                all_candidates.insert(all_candidates.end(), 
                                     thread_candidates[t].begin(), 
                                     thread_candidates[t].end());
            }
            
            if (verbosity >= 2) {
                std::cout << "[PoolBuild] Stage 1 collected " << all_candidates.size() 
                          << " candidates (need " << (target_size - pool_set.size()) << ")" << std::endl;
            }
            
            // Sort by heat_bath score for pre-filtering
            std::sort(all_candidates.begin(), all_candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                         return a.screening_score > b.screening_score;
                     });
            
            // Take top effective_target candidates by heat_bath score
            size_t pre_filter_count = std::min(all_candidates.size(), effective_target);
            
            if (verbosity >= 2) {
                std::cout << "[PoolBuild] Stage 2: computing PT2 for top " << pre_filter_count 
                          << " candidates" << std::endl;
            }
            
            // Compute H_jj and PT2 score for pre-filtered candidates
            for (size_t i = 0; i < pre_filter_count; ++i) {
                auto& cand = all_candidates[i];
                // Compute H_jj (diagonal element of child)
                cand.h_jj = compute_H_ij_t(cand.child, cand.child, h1, eri);
                double denom = e0 - cand.h_jj;
                
                // Avoid division by zero for intruder states
                if (std::abs(denom) < pt2_denom_min) {
                    denom = (denom >= 0) ? pt2_denom_min : -pt2_denom_min;
                }
                
                // PT2 score = |H_ij * c_i|^2 / |E_0 - H_jj|
                double hij_ci = cand.hij * cand.est_cj;
                cand.screening_score = std::abs((hij_ci * hij_ci) / denom);
            }
            
            // Re-sort pre-filtered candidates by PT2 score
            auto pt2_candidates_begin = all_candidates.begin();
            auto pt2_candidates_end = all_candidates.begin() + pre_filter_count;
            std::sort(pt2_candidates_begin, pt2_candidates_end,
                     [](const Candidate& a, const Candidate& b) {
                         return a.screening_score > b.screening_score;
                     });
            
            // For full PT2 mode: aggregate contributions from multiple parents
            if (use_pt2_aggregation) {
                std::unordered_map<DeterminantT<StorageType>, double> child_aggregated_score;
                std::unordered_map<DeterminantT<StorageType>, DeterminantT<StorageType>> child_best_parent;
                std::unordered_map<DeterminantT<StorageType>, double> child_best_hij;
                std::unordered_map<DeterminantT<StorageType>, double> child_best_score;

                for (size_t i = 0; i < pre_filter_count; ++i) {
                    const auto& cand = all_candidates[i];
                    auto it = child_aggregated_score.find(cand.child);
                    if (it == child_aggregated_score.end()) {
                        child_aggregated_score[cand.child] = cand.screening_score;
                        child_best_parent[cand.child] = cand.parent;
                        child_best_hij[cand.child] = cand.hij;
                        child_best_score[cand.child] = cand.screening_score;
                    } else {
                        it->second += cand.screening_score;
                        if (cand.screening_score > child_best_score[cand.child]) {
                            child_best_score[cand.child] = cand.screening_score;
                            child_best_parent[cand.child] = cand.parent;
                            child_best_hij[cand.child] = cand.hij;
                        }
                    }
                }
                
                // Sort by aggregated score
                std::vector<std::pair<DeterminantT<StorageType>, double>> sorted_children;
                sorted_children.reserve(child_aggregated_score.size());
                for (const auto& kv : child_aggregated_score) {
                    sorted_children.emplace_back(kv.first, kv.second);
                }
                std::sort(sorted_children.begin(), sorted_children.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
                
                for (const auto& [child, agg_score] : sorted_children) {
                    if (pool_set.insert(child).second) {
                        cache[pair_key_t(child_best_parent[child], child)] = child_best_hij[child];
                        new_frontier.push_back(child);
                        if (use_coeffs) {
                            coeff_map[child] = agg_score;
                        }
                    }
                }
                // Check target after processing all aggregated candidates
                if (pool_set.size() >= target_size) {
                    reached.store(true);
                }
            } else {
                // heat_bath_pt2: add pre-filtered candidates in PT2 score order
                for (size_t i = 0; i < pre_filter_count; ++i) {
                    const auto& cand = all_candidates[i];
                    if (pool_set.insert(cand.child).second) {
                        cache[pair_key_t(cand.parent, cand.child)] = cand.hij;
                        new_frontier.push_back(cand.child);
                        if (use_coeffs) {
                            coeff_map[cand.child] = cand.est_cj;
                        }
                    }
                }
                // Check target after processing all pre-filtered candidates
                if (pool_set.size() >= target_size) {
                    reached.store(true);
                }
            }
        } else {
            // heat_bath mode: Process all threads then check target (deterministic)
            for (int t = 0; t < n_threads; ++t) {
                for (const auto& cand : thread_candidates[t]) {
                    if (pool_set.insert(cand.child).second) {
                        cache[pair_key_t(cand.parent, cand.child)] = cand.hij;
                        new_frontier.push_back(cand.child);
                        if (use_coeffs) {
                            coeff_map[cand.child] = cand.est_cj;
                        }
                    }
                }
            }
            // Check target after all threads merged
            if (pool_set.size() >= target_size) {
                reached.store(true);
            }
        }

        if (reached.load()) {
            if (verbosity >= 1) {
                std::cout << "[PoolBuild] target_size reached, stopping.\n";
            }
            frontier.swap(new_frontier);
            break;
        }

        frontier.swap(new_frontier);
        ++round;
    }

    if (verbosity >= 1) {
        std::cout << "[PoolBuild] Final pool size: " << pool_set.size() 
              << ", final threshold: " << threshold << std::endl;
    }

    std::vector<DeterminantT<StorageType>> final_pool(pool_set.begin(), pool_set.end());
    
    // Strict target size truncation
    if (params.strict_target_size && final_pool.size() > target_size) {
        // Sort by coefficient (importance) if available, otherwise keep as-is
        if (use_coeffs) {
            std::sort(final_pool.begin(), final_pool.end(),
                     [&coeff_map](const DeterminantT<StorageType>& a, const DeterminantT<StorageType>& b) {
                         double ca = 0.0, cb = 0.0;
                         auto it_a = coeff_map.find(a);
                         auto it_b = coeff_map.find(b);
                         if (it_a != coeff_map.end()) ca = std::abs(it_a->second);
                         if (it_b != coeff_map.end()) cb = std::abs(it_b->second);
                         return ca > cb;
                     });
        }
        final_pool.resize(target_size);
        if (verbosity >= 1) {
            std::cout << "[PoolBuild] Strict truncation: " << pool_set.size() 
                      << " -> " << target_size << std::endl;
        }
    }
    
    return {final_pool, threshold};
}

// Wrapper for backward compatibility (pool_build)
// Note: uses heat_bath mode (default) for backward compatibility
std::pair<std::vector<Determinant>, double>
pool_build(
    const std::vector<Determinant>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCache& cache,
    const std::string& cache_file,
    int max_rounds,
    double threshold_decay,
    const std::vector<int>& attentive_orbitals,
    int verbosity
) {
    // Build params for backward compatibility
    PoolBuildParams params;
    params.max_rounds = max_rounds;
    params.threshold_decay = threshold_decay;
    // screening_mode defaults to "heat_bath"
    return pool_build_t<uint64_t>(initial_pool, initial_coeff, n_orb, h1, eri, 
                                  threshold, target_size, cache, cache_file, 
                                  attentive_orbitals, verbosity, params);
}

// Explicit instantiations
template std::vector<std::pair<DeterminantT<uint64_t>, double>>
process_parent_worker_t<uint64_t>(const DeterminantT<uint64_t>&, int, double, const DoubleExcTable&,
                                  const std::vector<std::vector<double>>&,
                                  const std::vector<double>&,
                                  const std::unordered_set<int>&,
                                  const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 2>>, double>>
process_parent_worker_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 3>>, double>>
process_parent_worker_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);

template std::pair<std::vector<DeterminantT<uint64_t>>, double>
pool_build_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, const std::vector<double>&, int,
                       const std::vector<std::vector<double>>&,
                       const std::vector<double>&,
                       double, size_t, HijCacheT<uint64_t>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 2>>>, double>
pool_build_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 2>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 3>>>, double>
pool_build_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 3>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);

template std::vector<std::pair<DeterminantT<std::array<uint64_t, 4>>, double>>
process_parent_worker_t<std::array<uint64_t, 4>>(const DeterminantT<std::array<uint64_t, 4>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 5>>, double>>
process_parent_worker_t<std::array<uint64_t, 5>>(const DeterminantT<std::array<uint64_t, 5>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 6>>, double>>
process_parent_worker_t<std::array<uint64_t, 6>>(const DeterminantT<std::array<uint64_t, 6>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 7>>, double>>
process_parent_worker_t<std::array<uint64_t, 7>>(const DeterminantT<std::array<uint64_t, 7>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 8>>, double>>
process_parent_worker_t<std::array<uint64_t, 8>>(const DeterminantT<std::array<uint64_t, 8>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);

template std::pair<std::vector<DeterminantT<std::array<uint64_t, 4>>>, double>
pool_build_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 4>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 5>>>, double>
pool_build_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 5>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 6>>>, double>
pool_build_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 6>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 7>>>, double>
pool_build_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 7>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 8>>>, double>
pool_build_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, const std::vector<double>&, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 8>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&);

} // namespace trimci_core
