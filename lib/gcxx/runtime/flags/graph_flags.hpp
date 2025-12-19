#pragma once
#ifndef GCXX_RUNTIME_FLAGS_GRAPH_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_GRAPH_FLAGS_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

enum class graphCreate : flag_t {
  none = 0  // as per cuda decumentation they may make new flags in future so
            // for now just set this
};


GCXX_NAMESPACE_MAIN_FLAGS_END

#endif
