#ifndef NETWORK_RESULT_H_INCLUDED
#define NETWORK_RESULT_H_INCLUDED

#include <utility>
#include <vector>

using scored_node = std::pair<float, int>;
using Netresult = std::pair<std::vector<scored_node>, float>;

#endif