// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_OPERATIONS_OPERATIONS_HPP_
#define GCXX_BLAS_OPERATIONS_OPERATIONS_HPP_

#include <gcxx/internal/prologue.hpp>

// BLAS type map (typed-op dispatch) + data-type trait, shared across all
// levels.
#include <gcxx/blas/datatypes/datatypes.hpp>
#include <gcxx/blas/type_map.hpp>

// Shared (non-level-specific) view helpers.
#include <gcxx/blas/operations/scaled.hpp>
#include <gcxx/blas/operations/side.hpp>
#include <gcxx/blas/operations/symmetric.hpp>
#include <gcxx/blas/operations/transposed.hpp>

// Per-level barrels.
#include <gcxx/blas/operations/L1/L1.hpp>
#include <gcxx/blas/operations/L2/L2.hpp>
#include <gcxx/blas/operations/L3/L3.hpp>

#endif
