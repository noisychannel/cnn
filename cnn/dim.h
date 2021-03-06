#ifndef CNN_DIM_H
#define CNN_DIM_H

#include <initializer_list>
#include <type_traits>
#include <iosfwd>
#include <cstring>

#define CNN_MAX_TENSOR_DIM 7

namespace boost { namespace serialization { class access; } }

namespace cnn {

struct Dim {
  Dim() : nd() {}
  explicit Dim(int m) : nd(1) { d[0] = m; }
  Dim(int m, int n) : nd(2) { d[0] = m; d[1] = n; }
  Dim(std::initializer_list<long> x) : nd() {
    for(auto v : x) d[nd++] = v;
  }
  inline int size() const {
    int p = 1;
    for (unsigned i = 0; i < nd; ++i) p *= d[i];
    return p;
  }
  inline int ndims() const { return nd; }
  inline int rows() const { return d[0]; }
  inline int cols() const { return nd > 1 ? d[1] : 1; }
  inline int operator[](unsigned i) const { return i < nd ? d[i] : 1; }
  inline int size(unsigned i) const { return (*this)[i]; }
  inline Dim transpose() const { return Dim(d[1],d[0]); }
  unsigned short d[CNN_MAX_TENSOR_DIM];
  unsigned short nd;
 private:
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & nd;
    ar & d;
  }
};

//static_assert(std::is_trivially_copyable<Dim>::value, "Dim must be trivially copyable");

inline bool operator==(const Dim& a, const Dim& b) {
  if (a.nd != b.nd) return false;
  return std::memcmp(a.d, b.d, a.nd) == 0;
}

inline bool operator!=(const Dim& a, const Dim& b) { return !(a == b); }

std::ostream& operator<<(std::ostream& os, const Dim& d);

} // namespace cnn

#endif
