// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_OPERATIONS_HPP_
#define GCXX_BLAS_OPERATIONS_OPERATIONS_HPP_

#include <gcxx/internal/prologue.hpp>

// BLAS type map (typed-op dispatch) + data-type trait shared by all levels.
#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/type_map.hpp>

// Shared (non-level-specific) helpers: scaled()/transposed() views + tags.
#include <gcxx/blas/operations/diagonal.hpp>
#include <gcxx/blas/operations/side.hpp>
#include <gcxx/blas/operations/symmetric.hpp>

<<<<<<< HEAD
// Per-level barrels.
#include <gcxx/blas/operations/L1/L1.hpp>
=======
// Per-level barrels. Future: L1.hpp (vector-vector).
>>>>>>> f6989c9 (Amending to new examples)
#include <gcxx/blas/operations/L2/L2.hpp>
#include <gcxx/blas/operations/L3/L3.hpp>

#endif
