// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_INTERNAL_PROLOGUE_HPP_
#define GCXX_INTERNAL_PROLOGUE_HPP_

#include <gcxx/macros/backend_mode_macros.hpp>

#if !GCXX_CPU_MODE()
#include <gcxx/backend/backend.hpp>
#endif

#include <gcxx/macros/function_decorator_macros.hpp>
#include <gcxx/macros/namespace_macros.hpp>
#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/macros/version_macros.hpp>

#include <cassert>

#ifndef GCXX_RUNTIME_EXPECT
#define GCXX_RUNTIME_EXPECT(COND, MSG) assert((COND) && (MSG))
#endif

#ifndef GCXX_STATIC_EXPECT
#define GCXX_STATIC_EXPECT(COND, MSG) static_assert(COND, MSG)
#endif

#if !GCXX_CPU_MODE()
#include <gcxx/runtime/runtime_error.hpp>
#endif


GCXX_NAMESPACE_MAIN_BEGIN()
using device_t = int;
GCXX_NAMESPACE_MAIN_END()

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()
using flag_t = unsigned int;
GCXX_NAMESPACE_MAIN_DETAILS_END()

#endif