#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <mudock/molecule.hpp>
#include <mudock/molecule/properties.hpp>
#include <string>

namespace mudock {

  // Serial sink filter for the phase0 pipeline.
  // Writes "name score\n" to stdout, buffering every 4096 lines.
  // Holds references to an external buffer and line counter so that the
  // caller can flush any residual output after parallel_pipeline returns.
  class writer_filter {
    std::string& buf;
    std::size_t& num_lines;

  public:
    writer_filter(std::string& buf_, std::size_t& num_lines_): buf(buf_), num_lines(num_lines_) {}

    void operator()(std::unique_ptr<static_molecule> scored_ligand) const {
      if (!scored_ligand)
        return;

      buf += scored_ligand->properties.get(property_type::NAME);
      buf += ' ';
      buf += scored_ligand->properties.get(property_type::SCORE);
      buf += '\n';

      if (++num_lines % 4096 == 0) {
        std::cout << buf;
        buf.clear();
      }
    }
  };

} // namespace mudock
