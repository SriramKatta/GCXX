#pragma once


#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gcxx/api.hpp>

struct Args {
  size_t N{};
  size_t rep{};
  size_t blocks{};
  size_t threads{};
};

inline Args parse_args(int argc, char** argv) {
  argparse::ArgumentParser program("vector_add");

  program.add_argument("-N", "--num-entries")
    .help("Number of elements")
    .default_value<size_t>(32'000'000)
    .scan<'i', std::size_t>();

  program.add_argument("-R", "--reps")
    .help("Number of kernel repetitions")
    .default_value<size_t>(10)
    .scan<'i', std::size_t>();

  program.add_argument("-B", "--blocks")
    .help("Number of blocks")
    .default_value<size_t>(3456)
    .scan<'i', std::size_t>();

  program.add_argument("-T", "--threads")
    .help("Threads per block")
    .default_value<size_t>(256)
    .scan<'i', std::size_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    fmt::print(stderr, "{}\n", err.what());
    fmt::print(stderr, "{}\n", program.help().str());
    std::exit(1);
  }

  return {program.get<size_t>("N"), program.get<size_t>("reps"),
          program.get<size_t>("blocks"), program.get<size_t>("threads")};
}

void launch_scalar_kernel(const Args& arg, const gcxx::Stream& str,
                          gcxx::span<double>&);

void launch_vec2_kernel(const Args& arg, const gcxx::Stream& str,
                        gcxx::span<double>&);

void launch_vec4_kernel(const Args& arg, const gcxx::Stream& str,
                        gcxx::span<double>&);