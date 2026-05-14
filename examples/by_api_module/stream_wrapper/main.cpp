// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#include <gcxx/runtime/stream.hpp>
#include <iostream>

int main() {
  namespace gflags = gcxx::flags;
  gcxx::Stream str1;
  gcxx::Stream str2(gflags::streamType::NoSyncWithNull);
  auto str1default = gcxx::Stream(gflags::streamType::SyncWithNull);
  auto str3        = gcxx::Stream(gflags::streamType::NoSyncWithNull);
  auto str4        = gcxx::Stream(gflags::streamType::NullStream,
                                  gflags::streamPriority::VeryHigh);

  std::cout << "size of stream obj " << sizeof(str1) << std::endl;
  std::cout << "size of stream obj " << sizeof(str1.get()) << std::endl;
  return 0;
}
