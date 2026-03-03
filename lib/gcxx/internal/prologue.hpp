#pragma once
#ifndef GCXX_INTERNAL_PROLOGUE_HPP_
#define GCXX_INTERNAL_PROLOGUE_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/backend_version_macros.hpp>
#include <gcxx/macros/function_decorator_macros.hpp>
#include <gcxx/macros/namespace_macros.hpp>
#include <gcxx/macros/template_helper_macros.hpp>
#include <gcxx/runtime/runtime_error.hpp>


GCXX_NAMESPACE_MAIN_BEGIN
using device_t = int;
GCXX_NAMESPACE_MAIN_END

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using flag_t = unsigned int;
GCXX_NAMESPACE_MAIN_DETAILS_END

#endif