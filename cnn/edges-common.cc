#include "cnn/edges.h"

#include <limits>
#include <cmath>
#include <sstream>

using namespace std;

namespace cnn {

inline ostream& operator<<(ostream& os, const vector<Dim>& ds) {
  os << '[';
  for (unsigned i = 0; i < ds.size(); ++i)
    os << (i ? " " : "") << ds[i];
  return os << ']';
}

inline bool LooksLikeVector(const Dim& d) {
  if (d.ndims() == 1) return true;
  if (d.ndims() > 1) {
    for (unsigned i = 1; i < d.ndims(); ++i)
      if (d[i] != 1) return false;
  }
  return true;
}

string Reshape::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "reshape(" << arg_names[0] << ',' << from << " --> " << to << ')';
  return s.str();
}

Dim Reshape::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  assert(xs[0] == from);
  return to;
}

string SumColumns::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "sum_cols(" << arg_names[0] << ')';
  return s.str();
}

Dim SumColumns::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return Dim({xs[0].rows()});
}

string KMHNGram::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "kmh-ngram(" << arg_names[0] << ')';
  return s.str();
}

Dim KMHNGram::dim_forward(const vector<Dim>& xs) const {
  assert(xs[0].ndims() == 2);
  const int new_cols = xs[0].cols() - n + 1;
  if (new_cols < 1) {
    cerr << "Bad input dimensions in KMHNGram: " << xs << endl;
    abort();
  }
  return Dim({xs[0][0], new_cols});
}

string InnerProduct3D_1D::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "inner(" << arg_names[0] << "," << arg_names[1] << ") + " << arg_names[2];
  return s.str();
}

Dim InnerProduct3D_1D::dim_forward(const vector<Dim>& xs) const {
  cerr << "InnerProduct3D_1D::dim_forward not implemented\n";
  abort();
}

string GaussianNoise::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " + N(0," << stddev << ')';
  return s.str();
}

Dim GaussianNoise::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Dropout::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "dropout(" << arg_names[0] << ",p=" << p << ')';
  return s.str();
}

Dim Dropout::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string ConstantMinusX::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << c << " - " << arg_names[0];
  return s.str();
}

Dim ConstantMinusX::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Sum::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i)
    s << " + " << arg_names[i];
  return s.str();
}

Dim Sum::dim_forward(const vector<Dim>& xs) const {
  for (unsigned i = 1; i < xs.size(); ++i) {
    if (xs[0] != xs[1]) {
      cerr << "Mismatched input dimensions in Sum: " << xs << endl;
      abort();
    }
  }
  return xs[0];
}

string Tanh::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "tanh(" << arg_names[0] << ')';
  return s.str();
}

Dim Tanh::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Square::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "square(" << arg_names[0] << ')';
  return s.str();
}

Dim Square::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Exp::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "exp(" << arg_names[0] << ')';
  return os.str();
}

Dim Exp::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Log::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "log(" << arg_names[0] << ')';
  return os.str();
}

Dim Log::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Concatenate::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Dim Concatenate::dim_forward(const vector<Dim>& xs) const {
  unsigned new_rows = 0;
  for (auto& d : xs) {
    if (!LooksLikeVector(d)) {
      cerr << "Bad input dimensions in Concatenate: " << xs << endl;
      abort();
    }
    new_rows += d[0];
  }
  return Dim({new_rows});
}

string ConcatenateColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "concat_cols(" << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); ++i) {
    os << ',' << arg_names[i];
  }
  os << ')';
  return os.str();
}

Dim ConcatenateColumns::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() > 0);
  unsigned rows = xs[0][0];
  unsigned new_cols = 0;
  for (auto& d : xs) {
    if (d[0] != rows) {
      cerr << "Bad input dimensions in ConcatenateColumns: " << xs << endl;
      abort();
    }
    new_cols += d[1];
  }
  return Dim({rows, new_cols});
}

string Hinge::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "hinge(" << arg_names[0] << ",m=" << margin << ")";
  return os.str();
}

Dim Hinge::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in Hinge: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string Identity::as_string(const vector<string>& arg_names) const {
  return arg_names[0];
}

Dim Identity::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string MaxPooling1D::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "maxpool1d(" << arg_names.front() << ",w=" << width << ")";
  return os.str();
}

Dim MaxPooling1D::dim_forward(const vector<Dim>& xs) const {
  cerr << "MaxPooling1D::dim_forward not implemented\n";
  abort();
}

string Softmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim Softmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in Softmax: " << xs << endl;
    abort();
  }
  return xs[0];
}

string PickNegLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ")_{" << *pval << '}';
  return s.str();
}

Dim PickNegLogSoftmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in PickNegLogSoftmax: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string LogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim LogSoftmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in LogSoftmax: " << xs << endl;
    abort();
  }
  return xs[0];
}

string RestrictedLogSoftmax::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "r_log_softmax(" << arg_names[0] << ')';
  return s.str();
}

Dim RestrictedLogSoftmax::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in RestrictedLogSoftmax: " << xs << endl;
    abort();
  }
  return xs[0];
}

string PickElement::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "pick(" << arg_names[0] << ',' << *pval << ')';
  return s.str();
}

Dim PickElement::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in PickElement: " << xs << endl;
    abort();
  }
  return Dim({1});
}

// x_1 is a vector
// y = (x_1)[start:end]
string PickRange::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "slice(" << arg_names[0] << ',' << start << ':' << end << ')';
  return s.str();
}

Dim PickRange::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (!LooksLikeVector(xs[0])) {
    cerr << "Bad input dimensions in PickElement: " << xs << endl;
    abort();
  }
  assert(xs[0][0] <= end);
  return Dim({end - start});
}

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Dim MatrixMultiply::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0].cols() != xs[1].rows()) {
    cerr << "Mismatched input dimensions in MatrixMultiply: " << xs << endl;
    abort();
  }
  if (xs[1].ndims() == 1) return Dim({xs[0].rows()});
  return Dim({xs[0].rows(), xs[1].cols()});
}

string CwiseMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " \\cdot " << arg_names[1];
  return s.str();
}

Dim CwiseMultiply::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in CwiseMultiply: " << xs << endl;
    abort();
  }
  return xs[0];
}

string Multilinear::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0];
  for (unsigned i = 1; i < arg_names.size(); i += 2)
    s << " + " << arg_names[i] << " * " << arg_names[i+1];
  return s.str();
}

Dim Multilinear::dim_forward(const vector<Dim>& xs) const {
  if ((xs.size() - 1) % 2 != 0) {
    cerr << "Bad number of inputs for Multilinear: " << xs << endl;
    abort();
  }
  for (unsigned i = 1; i < xs.size(); i += 2) {
    if (xs[i].cols() != xs[i+1].rows() ||
        xs[0].rows() != xs[i].rows() ||
        xs[0].cols() != xs[i+1].cols()) {
      cerr << "Bad dimensions for Multilinear: " << xs << endl;
      abort();
    }
  }
  return xs[0];
}

string Negate::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << '-' << arg_names[0];
  return s.str();
}

Dim Negate::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string Rectify::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "ReLU(" << arg_names[0] << ')';
  return s.str();
}

Dim Rectify::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string SquaredEuclideanDistance::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "|| " << arg_names[0] << " - " << arg_names[1] << " ||^2";
  return s.str();
}

Dim SquaredEuclideanDistance::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 2);
  if (xs[0] != xs[1]) {
    cerr << "Mismatched input dimensions in SquaredEuclideanDistance: " << xs << endl;
    abort();
  }
  return Dim({1});
}

string LogisticSigmoid::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "\\sigma(" << arg_names[0] << ')';
  return s.str();
}

Dim LogisticSigmoid::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  return xs[0];
}

string BinaryLogLoss::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "binary_log_loss(" << arg_names[0] << ", " << *ptarget_y << ')';
  return os.str();
}

Dim BinaryLogLoss::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 1);
  if (xs[0].rows() != 2 && xs[0].ndims() != 1) {
    cerr << "Bad input dimensions in BinaryLogLoss: " << xs << endl;
    abort();
  }
  return Dim({1});
}

} // namespace cnn
