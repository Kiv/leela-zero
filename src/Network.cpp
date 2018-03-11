/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "config.h"
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"

#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"
#include "NetworkBase.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;
using namespace NetworkBase;

// Input + residual block tower
static std::vector<std::vector<float>> conv_weights;
static std::vector<std::vector<float>> conv_biases;
static std::vector<std::vector<float>> batchnorm_means;
static std::vector<std::vector<float>> batchnorm_stddivs;

// Policy head
static std::vector<float> conv_pol_w;
static std::vector<float> conv_pol_b;
static std::array<float, 2> bn_pol_w1;
static std::array<float, 2> bn_pol_w2;

static std::array<float, (BOARD_SQUARES + 1) * BOARD_SQUARES * 2> ip_pol_w;
static std::array<float, BOARD_SQUARES + 1> ip_pol_b;

// Value head
static std::vector<float> conv_val_w;
static std::vector<float> conv_val_b;
static std::array<float, 1> bn_val_w1;
static std::array<float, 1> bn_val_w2;

static std::array<float, BOARD_SQUARES * 256> ip1_val_w;
static std::array<float, 256> ip1_val_b;

static std::array<float, 256> ip2_val_w;
static std::array<float, 1> ip2_val_b;

// Rotation helper
static std::array<std::array<int, BOARD_SQUARES>, 8> rotate_nn_idx_table;

void Network::benchmark(const GameState * state, int iterations) {
    int cpus = cfg_num_threads;
    int iters_per_thread = (iterations + (cpus - 1)) / cpus;

    Time start;

    ThreadGroup tg(thread_pool);
    for (int i = 0; i < cpus; i++) {
        tg.add_task([iters_per_thread, state]() {
            for (int loop = 0; loop < iters_per_thread; loop++) {
                auto vec = get_scored_moves(state, Ensemble::RANDOM_ROTATION, -1, true);
            }
        });
    };
    tg.wait_all();

    Time end;
    auto elapsed = Time::timediff_seconds(start,end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             iterations, elapsed, (int)(iterations / elapsed));
}

std::pair<int, int>  Network::load_v1_network(std::ifstream& wtfile) {
    // Count size of the network
    myprintf("Detecting residual layers...");
    // We are version 1
    myprintf("v%d...", 1);
    // First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    auto line = std::string{};
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
            channels = count;
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        return {0, 0};
    }
    residual_blocks /= 8;
    myprintf("%d blocks.\n", residual_blocks);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    auto plain_conv_layers = 1 + (residual_blocks * 2);
    auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.begin();
        auto ok = phrase_parse(it_line, line.end(),
                               *x3::float_, x3::space, weights);
        if (!ok || it_line != line.end()) {
            myprintf("\nFailed to parse weight file. Error on line %d.\n",
                    linecount + 2); //+1 from version line, +1 from 0-indexing
            return {0,0};
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                batchnorm_stddivs.emplace_back(weights);
            }
        } else if (linecount == plain_conv_wts) {
            conv_pol_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 1) {
            conv_pol_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 2) {
            std::copy(begin(weights), end(weights), begin(bn_pol_w1));
        } else if (linecount == plain_conv_wts + 3) {
            process_bn_var(weights);
            std::copy(begin(weights), end(weights), begin(bn_pol_w2));
        } else if (linecount == plain_conv_wts + 4) {
            std::copy(begin(weights), end(weights), begin(ip_pol_w));
        } else if (linecount == plain_conv_wts + 5) {
            std::copy(begin(weights), end(weights), begin(ip_pol_b));
        } else if (linecount == plain_conv_wts + 6) {
            conv_val_w = std::move(weights);
        } else if (linecount == plain_conv_wts + 7) {
            conv_val_b = std::move(weights);
        } else if (linecount == plain_conv_wts + 8) {
            std::copy(begin(weights), end(weights), begin(bn_val_w1));
        } else if (linecount == plain_conv_wts + 9) {
            process_bn_var(weights);
            std::copy(begin(weights), end(weights), begin(bn_val_w2));
        } else if (linecount == plain_conv_wts + 10) {
            std::copy(begin(weights), end(weights), begin(ip1_val_w));
        } else if (linecount == plain_conv_wts + 11) {
            std::copy(begin(weights), end(weights), begin(ip1_val_b));
        } else if (linecount == plain_conv_wts + 12) {
            std::copy(begin(weights), end(weights), begin(ip2_val_w));
        } else if (linecount == plain_conv_wts + 13) {
            std::copy(begin(weights), end(weights), begin(ip2_val_b));
        }
        linecount++;
    }
    wtfile.close();

    return {channels, residual_blocks};
}

std::pair<int, int> Network::load_network_file(std::string filename) {
    auto wtfile = std::ifstream{filename};
    if (wtfile.fail()) {
        myprintf("Could not open weights file: %s\n", filename.c_str());
        return {0, 0};
    }

    // Read format version
    auto line = std::string{};
    auto format_version = -1;
    if (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // First line is the file format version id
        iss >> format_version;
        if (iss.fail() || format_version != FORMAT_VERSION) {
            myprintf("Weights file is the wrong version.\n");
            return {0, 0};
        } else {
            assert(format_version == FORMAT_VERSION);
            return load_v1_network(wtfile);
        }
    }

    return {0, 0};
}

void Network::initialize(void) {
    // Prepare rotation table
    for(auto s = 0; s < 8; s++) {
        for(auto v = 0; v < BOARD_SQUARES; v++) {
            rotate_nn_idx_table[s][v] = rotate_nn_idx(v, s);
        }
    }

    // Load network from file
    size_t channels, residual_blocks;
    std::tie(channels, residual_blocks) = load_network_file(cfg_weightsfile);
    if (channels == 0) {
        exit(EXIT_FAILURE);
    }

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    conv_weights[weight_index] =
        winograd_transform_f(conv_weights[weight_index],
                             channels, INPUT_CHANNELS);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < residual_blocks * 2; i++) {
		conv_weights[weight_index] =
            winograd_transform_f(conv_weights[weight_index],
                                 channels, channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    for (auto i = size_t{0}; i < conv_biases.size(); i++) {
        for (auto j = size_t{0}; j < batchnorm_means[i].size(); j++) {
            batchnorm_means[i][j] -= conv_biases[i][j];
            conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i < bn_val_w1.size(); i++) {
        bn_val_w1[i] -= conv_val_b[i];
        conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < bn_pol_w1.size(); i++) {
        bn_pol_w1[i] -= conv_pol_b[i];
        conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    myprintf("Initializing OpenCL.\n");
    opencl.initialize(channels);

    for(auto & opencl_net : opencl.get_networks()) {
        auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        auto mwg = tuners[0];
        auto kwg = tuners[2];
        auto vwm = tuners[3];

        weight_index = 0;

        size_t m_ceil = ceilMultiple(ceilMultiple(channels, mwg), vwm);
        size_t k_ceil = ceilMultiple(ceilMultiple(INPUT_CHANNELS, kwg), vwm);

        auto Upad = zeropad_U(conv_weights[weight_index],
                              channels, INPUT_CHANNELS,
                              m_ceil, k_ceil);

        // Winograd filter transformation changes filter size to 4x4
        opencl_net->push_input_convolution(WINOGRAD_ALPHA, INPUT_CHANNELS, channels,
                Upad, batchnorm_means[weight_index], batchnorm_stddivs[weight_index]);
        weight_index++;

        // residual blocks
        for (auto i = size_t{0}; i < residual_blocks; i++) {
            auto Upad1 = zeropad_U(conv_weights[weight_index],
                                   channels, channels,
                                   m_ceil, m_ceil);
            auto Upad2 = zeropad_U(conv_weights[weight_index + 1],
                                   channels, channels,
                                   m_ceil, m_ceil);
            opencl_net->push_residual(WINOGRAD_ALPHA, channels, channels,
                                      Upad1,
                                      batchnorm_means[weight_index],
                                      batchnorm_stddivs[weight_index],
                                      Upad2,
                                      batchnorm_means[weight_index + 1],
                                      batchnorm_stddivs[weight_index + 1]);
            weight_index += 2;
        }

        // Output head convolutions
        opencl_net->push_convolve1(channels, OUTPUTS_POLICY, conv_pol_w);
        opencl_net->push_convolve1(channels, OUTPUTS_VALUE, conv_val_w);
    }
#endif
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    //mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#endif
}

#ifdef USE_BLAS
void Network::forward_cpu(std::vector<float>& input,
                          std::vector<float>& output_pol,
                          std::vector<float>& output_val) {
    // Input convolution
    constexpr int width = BOARD_SIZE;
    constexpr int height = BOARD_SIZE;
    constexpr int tiles = (width + 1) * (height + 1) / 4;
    // Calculate output channels
    const auto output_channels = conv_biases[0].size();
    //input_channels is the maximum number of input channels of any convolution.
    //Residual blocks are identical, but the first convolution might be bigger
    //when the network has very few filters
    const auto input_channels = std::max(
            static_cast<size_t>(output_channels),
            static_cast<size_t>(INPUT_CHANNELS));
    auto conv_out = std::vector<float>(output_channels * width * height);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * tiles);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * tiles);

    winograd_convolve3(output_channels, input, conv_weights[0], V, M, conv_out);
    batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                             batchnorm_means[0].data(),
                             batchnorm_stddivs[0].data());

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * width * height);
    auto res = std::vector<float>(output_channels * width * height);
    for (auto i = size_t{1}; i < conv_weights.size(); i += 2) {
        auto output_channels = conv_biases[i].size();
        std::swap(conv_out, conv_in);
        std::copy(begin(conv_in), end(conv_in), begin(res));
        winograd_convolve3(output_channels, conv_in,
	                       conv_weights[i], V, M, conv_out);
        batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                                 batchnorm_means[i].data(),
                                 batchnorm_stddivs[i].data());

        output_channels = conv_biases[i + 1].size();
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
			               conv_weights[i + 1], V, M, conv_out);
        batchnorm<BOARD_SQUARES>(output_channels, conv_out,
                                 batchnorm_means[i + 1].data(),
                                 batchnorm_stddivs[i + 1].data(),
                                 res.data());
    }
    convolve<1, height, width>(OUTPUTS_POLICY, conv_out, conv_pol_w, conv_pol_b, output_pol);
    convolve<1, height, width>(OUTPUTS_VALUE, conv_out, conv_val_w, conv_val_b, output_val);
}
#endif

Netresult Network::get_scored_moves(
    const GameState* state, Ensemble ensemble, int rotation, bool skip_cache) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    // See if we already have this in the cache.
    if (!skip_cache) {
      if (NNCache::get_NNCache().lookup(state->board.get_hash(), result)) {
        return result;
      }
    }

    NNPlanes planes;
    gather_features(state, planes);

    if (ensemble == DIRECT) {
        assert(rotation >= 0 && rotation <= 7);
        result = get_scored_moves_internal(state, planes, rotation);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        auto rand_rot = Random::get_Rng().randfix<8>();
        result = get_scored_moves_internal(state, planes, rand_rot);
    }

    // Insert result into cache.
    NNCache::get_NNCache().insert(state->board.get_hash(), result);

    return result;
}

Netresult Network::get_scored_moves_internal(
    const GameState* state, NNPlanes & planes, int rotation) {
    assert(rotation >= 0 && rotation <= 7);
    assert(INPUT_CHANNELS == planes.size());
    constexpr int width = BOARD_SIZE;
    constexpr int height = BOARD_SIZE;
    const auto convolve_channels = conv_pol_w.size() / conv_pol_b.size();
    std::vector<net_t> input_data;
    std::vector<net_t> output_data(convolve_channels * width * height);
    std::vector<float> policy_data(OUTPUTS_POLICY * width * height);
    std::vector<float> value_data(OUTPUTS_VALUE * width * height);
    std::vector<float> policy_out((width * height) + 1);
    std::vector<float> softmax_data((width * height) + 1);
    std::vector<float> winrate_data(256);
    std::vector<float> winrate_out(1);
    // Data layout is input_data[(c * height + h) * width + w]
    input_data.reserve(INPUT_CHANNELS * width * height);
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                auto rot_idx = rotate_nn_idx_table[rotation][h * BOARD_SIZE + w];
                input_data.emplace_back(net_t(planes[c][rot_idx]));
            }
        }
    }
#ifdef USE_OPENCL
    opencl.forward(input_data, policy_data, value_data);
#elif defined(USE_BLAS) && !defined(USE_OPENCL)
    forward_cpu(input_data, policy_data, value_data);
#endif
#ifdef USE_OPENCL_SELFCHECK
    // Both implementations are available, self-check the OpenCL driver by
    // running both with a probability of 1/2000.
    if (Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0) {
        auto cpu_policy_data = std::vector<float>(policy_data.size());
        auto cpu_value_data = std::vector<float>(value_data.size());
        forward_cpu(input_data, cpu_policy_data, cpu_value_data);
        compare_net_outputs(policy_data, cpu_policy_data);
        compare_net_outputs(value_data, cpu_value_data);
    }
#endif

    // Get the moves
    batchnorm<BOARD_SQUARES>(OUTPUTS_POLICY, policy_data, bn_pol_w1.data(), bn_pol_w2.data());
    innerproduct<OUTPUTS_POLICY * BOARD_SQUARES, BOARD_SQUARES + 1>(policy_data, ip_pol_w, ip_pol_b, policy_out);
    softmax(policy_out, softmax_data, cfg_softmax_temp);
    std::vector<float>& outputs = softmax_data;

    // Now get the score
    batchnorm<BOARD_SQUARES>(OUTPUTS_VALUE, value_data, bn_val_w1.data(), bn_val_w2.data());
    innerproduct<BOARD_SQUARES, 256>(value_data, ip1_val_w, ip1_val_b, winrate_data);
    innerproduct<256, 1>(winrate_data, ip2_val_w, ip2_val_b, winrate_out);

    // Sigmoid
    auto winrate_sig = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    std::vector<scored_node> result;
    for (auto idx = size_t{0}; idx < outputs.size(); idx++) {
        if (idx < BOARD_SQUARES) {
            auto val = outputs[idx];
            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            auto x = rot_idx % BOARD_SIZE;
            auto y = rot_idx / BOARD_SIZE;
            auto rot_vtx = state->board.get_vertex(x, y);
            if (state->board.get_square(rot_vtx) == FastBoard::EMPTY) {
                result.emplace_back(val, rot_vtx);
            }
        } else {
            result.emplace_back(outputs[idx], FastBoard::PASS);
        }
    }

    return std::make_pair(result, winrate_sig);
}

void Network::show_heatmap(const FastState * state, Netresult& result, bool topmoves) {
    auto moves = result.first;
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            int vtx = state->board.get_vertex(x, y);

            auto item = std::find_if(moves.cbegin(), moves.cend(),
                [&vtx](scored_node const& test_item) {
                return test_item.second == vtx;
            });

            auto score = 0.0f;
            // Non-empty squares won't be scored
            if (item != moves.end()) {
                score = item->first;
                assert(vtx == item->second);
            }

            line += boost::str(boost::format("%3d ") % int(score * 1000));
            if (x == BOARD_SIZE - 1) {
                display_map.push_back(line);
                line.clear();
            }
        }
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    assert(result.first.back().second == FastBoard::PASS);
    auto pass_score = int(result.first.back().first * 1000);
    myprintf("pass: %d\n", pass_score);
    myprintf("winrate: %f\n", result.second);

    if (topmoves) {
        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        size_t tried = 0;
        while (cum < 0.85f && tried < moves.size()) {
            if (moves[tried].first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    moves[tried].first,
                    state->board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    BoardPlane& black, BoardPlane& white) {
    auto idx = 0;
    for (int j = 0; j < BOARD_SIZE; j++) {
        for(int i = 0; i < BOARD_SIZE; i++) {
            int vtx = board.get_vertex(i, j);
            auto color = board.get_square(vtx);
            if (color != FastBoard::EMPTY) {
                if (color == FastBoard::BLACK) {
                    black[idx] = true;
                } else {
                    white[idx] = true;
                }
            }
            idx++;
        }
    }
}

void Network::gather_features(const GameState* state, NNPlanes & planes) {
    planes.resize(INPUT_CHANNELS);
    BoardPlane& black_to_move = planes[2 * INPUT_MOVES];
    BoardPlane& white_to_move = planes[2 * INPUT_MOVES + 1];

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    const auto black_offset = blacks_move ? 0 : INPUT_MOVES;
    const auto white_offset = blacks_move ? INPUT_MOVES : 0;

    if (blacks_move) {
        black_to_move.set();
    } else {
        white_to_move.set();
    }

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_board(h),
                              planes[black_offset + h],
                              planes[white_offset + h]);
    }
}

int Network::rotate_nn_idx(const int vertex, int symmetry) {
    assert(vertex >= 0 && vertex < BOARD_SQUARES);
    assert(symmetry >= 0 && symmetry < 8);
    int x = vertex % BOARD_SIZE;
    int y = vertex / BOARD_SIZE;
    int newx;
    int newy;

    if (symmetry >= 4) {
        std::swap(x, y);
        symmetry -= 4;
    }

    if (symmetry == 0) {
        newx = x;
        newy = y;
    } else if (symmetry == 1) {
        newx = x;
        newy = BOARD_SIZE - y - 1;
    } else if (symmetry == 2) {
        newx = BOARD_SIZE - x - 1;
        newy = y;
    } else {
        assert(symmetry == 3);
        newx = BOARD_SIZE - x - 1;
        newy = BOARD_SIZE - y - 1;
    }

    int newvtx = (newy * BOARD_SIZE) + newx;
    assert(newvtx >= 0 && newvtx < BOARD_SQUARES);
    return newvtx;
}
