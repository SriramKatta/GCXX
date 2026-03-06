#include <array>
#include <vector>

#include "tests_common.hpp"

#include <gcxx/api.hpp>

TEST(SpanConstructors, DataSizeCheck) {
#define HELPER_ELEMS \
  { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
  int carr[] = HELPER_ELEMS;
  std::array arr HELPER_ELEMS;
  const std::array const_arr HELPER_ELEMS;
  std::vector vec HELPER_ELEMS;
  gcxx::span<int> s1;
  gcxx::span s2{vec.begin(), vec.size()};
  gcxx::span s3{vec.begin(), vec.end()};
  gcxx::span s4{carr};
  gcxx::span s5{arr};
  gcxx::span s6{const_arr};
  gcxx::span s7(vec);
  // gcxx::span<float> s8{s7};
  gcxx::span s9(s7);

  EXPECT_EQ(s1.data(), nullptr) << "Default ctor data mismatch";
  EXPECT_EQ(s2.data(), vec.data()) << "iter, count ctor data mismatch";
  EXPECT_EQ(s3.data(), vec.data()) << "iter, iter ctor data mismatch";
  EXPECT_EQ(s4.data(), std::data(carr)) << "c_array ctor data mismatch";
  EXPECT_EQ(s5.data(), arr.data()) << "std::array ctor data mismatch";
  EXPECT_EQ(s6.data(), const_arr.data())
    << "const std::array ctor data mismatch";
  EXPECT_EQ(s7.data(), vec.data()) << "range ctor data mismatch";
  EXPECT_EQ(s9.data(), s7.data()) << "from another span ctor data mismatch";


  EXPECT_EQ(s1.size(), 0) << "Default ctor size mismatch";
  EXPECT_EQ(s2.size(), vec.size()) << "iter, count ctor size mismatch";
  EXPECT_EQ(s3.size(), vec.size()) << "iter, iter ctor size mismatch";
  EXPECT_EQ(s4.size(), std::size(carr)) << "c_array ctor size mismatch";
  EXPECT_EQ(s5.size(), arr.size()) << "std::array ctor size mismatch";
  EXPECT_EQ(s6.size(), const_arr.size())
    << "const std::array ctor size mismatch";
  EXPECT_EQ(s7.size(), vec.size()) << "range ctor size mismatch";
  EXPECT_EQ(s9.size(), s7.size()) << "from another span ctor size mismatch";
}
