#ifndef CNN_C2W_H_
#define CNN_C2W_H_

#include <vector>
#include <map>

#include "cnn/cnn.h"
#include "cnn/model.h"
#include "cnn/lstm.h"

namespace cnn {

// computes a representation of a word by reading characters
// one at a time
struct C2WBuilder {
  LSTMBuilder fc2w;
  LSTMBuilder rc2w;
  LookupParameters* p_lookup;
  std::vector<VariableIndex> words;
  std::map<int, VariableIndex> wordid2vi;
  explicit C2WBuilder(int vocab_size,
                      unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model* m) :
      fc2w(layers, input_dim, hidden_dim, m),
      rc2w(layers, input_dim, hidden_dim, m),
      p_lookup(m->add_lookup_parameters(vocab_size, {input_dim})) {
  }
  void new_graph(Hypergraph* hg) {
    words.clear();
    fc2w.new_graph(hg);
    rc2w.new_graph(hg);
  }
  // compute a composed representation of a word out of characters
  // wordid should be a unique index for each word *type* in the graph being built
  VariableIndex add_word(int word_id, const std::vector<int>& chars, Hypergraph* hg) {
    auto it = wordid2vi.find(word_id);
    if (it == wordid2vi.end()) {
      fc2w.start_new_sequence(hg);
      rc2w.start_new_sequence(hg);
      std::vector<VariableIndex> ins(chars.size());
      std::map<int, VariableIndex> c2i;
      for (unsigned i = 0; i < ins.size(); ++i) {
        VariableIndex& v = c2i[chars[i]];
        if (!v) v = hg->add_lookup(p_lookup, chars[i]);
        ins[i] = v;
        fc2w.add_input(v, hg);
      }
      for (int i = ins.size() - 1; i >= 0; --i)
        rc2w.add_input(ins[i], hg);
      VariableIndex i_concat = hg->add_function<Concatenate>({fc2w.back(), rc2w.back()});
      it = wordid2vi.insert(std::make_pair(word_id, i_concat)).first;
    }
    return it->second;
  }
};

} // namespace cnn

#endif
