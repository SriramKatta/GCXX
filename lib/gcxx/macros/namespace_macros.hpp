// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_MACROS_NAMESPACE_MACROS_HPP_
#define GCXX_MACROS_NAMESPACE_MACROS_HPP_

#ifndef GCXX_NAMESPACE_MAIN_BEGIN
#define GCXX_NAMESPACE_MAIN_BEGIN() \
  namespace gcxx {                  \
    inline namespace v1 {
#endif

#ifndef GCXX_NAMESPACE_MAIN_END
#define GCXX_NAMESPACE_MAIN_END() \
  } /* inline namespace v1  */    \
  }  // namespace gcxx
#endif

#ifndef GCXX_NAMESPACE_DETAILS_BEGIN
#define GCXX_NAMESPACE_DETAILS_BEGIN() namespace details_ {
#endif

#ifndef GCXX_NAMESPACE_DETAILS_END
#define GCXX_NAMESPACE_DETAILS_END() } /* namespace details_  */
#endif

#ifndef GCXX_NAMESPACE_DRIVER_BEGIN
#define GCXX_NAMESPACE_DRIVER_BEGIN() namespace driver {
#endif

#ifndef GCXX_NAMESPACE_DRIVER_END
#define GCXX_NAMESPACE_DRIVER_END() } /* namespace driver  */
#endif

#ifndef GCXX_NAMESPACE_FLAGS_BEGIN
#define GCXX_NAMESPACE_FLAGS_BEGIN() namespace flags {
#endif

#ifndef GCXX_NAMESPACE_FLAGS_END
#define GCXX_NAMESPACE_FLAGS_END() } /* namespace flags  */
#endif

#ifndef GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
#define GCXX_NAMESPACE_MAIN_DETAILS_BEGIN \
  GCXX_NAMESPACE_MAIN_BEGIN()             \
  GCXX_NAMESPACE_DETAILS_BEGIN()
#endif

#ifndef GCXX_NAMESPACE_MAIN_DETAILS_END
#define GCXX_NAMESPACE_MAIN_DETAILS_END \
  GCXX_NAMESPACE_DETAILS_END()          \
  GCXX_NAMESPACE_MAIN_END()
#endif

#ifndef GCXX_NAMESPACE_MAIN_FLAGS_BEGIN
#define GCXX_NAMESPACE_MAIN_FLAGS_BEGIN \
  GCXX_NAMESPACE_MAIN_BEGIN()           \
  GCXX_NAMESPACE_FLAGS_BEGIN()
#endif

#ifndef GCXX_NAMESPACE_MAIN_FLAGS_END
#define GCXX_NAMESPACE_MAIN_FLAGS_END \
  GCXX_NAMESPACE_FLAGS_END()          \
  GCXX_NAMESPACE_MAIN_END()
#endif

#ifndef GCXX_NAMESPACE_MAIN_DRIVER_BEGIN
#define GCXX_NAMESPACE_MAIN_DRIVER_BEGIN \
  GCXX_NAMESPACE_MAIN_BEGIN()            \
  GCXX_NAMESPACE_DRIVER_BEGIN()
#endif

#ifndef GCXX_NAMESPACE_MAIN_DRIVER_END
#define GCXX_NAMESPACE_MAIN_DRIVER_END \
  GCXX_NAMESPACE_DRIVER_END()          \
  GCXX_NAMESPACE_MAIN_END()
#endif


#endif
