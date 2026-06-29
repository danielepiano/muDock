#include "command_line_args.hpp"

#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mudock/compute/pipeline_selector.hpp>
#include <mudock/format/supported_format.hpp>
#include <mudock/molecule.hpp>
#include <mudock/mudock.hpp>
#include <mudock/tbb_implementation/tbb_pipeline_phase0.hpp>

int main(int argc, char** argv) {
  const auto args = parse_command_line_arguments(argc, argv);

  mudock::info("Reading and parsing protein ", args.protein_path, " ...");
  auto protein =
      std::make_shared<mudock::dynamic_molecule>(mudock::parser<mudock::dynamic_molecule>(args.protein_path));

  mudock::info("Reading ligand ", args.ligand_path, " ...");
  std::ifstream in(args.ligand_path, std::ios::binary);
  if (!in) {
    mudock::error("Can't open input file ", args.ligand_path);
    return EXIT_FAILURE;
  }

  const auto in_format = mudock::parse_supported_format(args.ligand_path);

  mudock::info("Pipeline selection (phase0): search=",
               to_string(args.search),
               ", score=",
               to_string(args.scoring));

  dispatch_selected_pipeline(args.search,
                             args.scoring,
                             in_format,
                             [&]<typename pipeline_t>(const auto, const auto) {
                               pipeline_t pipe{protein};
                               mudock::run_tbb_pipeline_phase0<mudock::supported_format::ADTMOL2>(
                                   in,
                                   args.device_confs,
                                   args.knobs,
                                   pipe,
                                   std::numeric_limits<std::size_t>::max(),
                                   args.time_limit_sec);
                             });

  mudock::info("All Done!");
  return EXIT_SUCCESS;
}
