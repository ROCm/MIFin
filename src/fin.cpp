#include "fin.hpp"

namespace fin {

[[gnu::noreturn]] void Fin::Usage() {
  std::cout << "Usage: ./MIOpenFin *base_arg* *other_args*\n";
  std::cout << "Supported Base Arguments: conv[fp16][bfp16]\n";
  exit(0);
}

std::string Fin::ParseBaseArg(const int argc, const char *argv[]) {
  if (argc < 2) {
    std::cout << "Invalid Number of Input Arguments\n";
    Usage();
  }

  std::string arg = argv[1];

  if (arg != "conv" && arg != "convfp16" && arg != "convbfp16") {
    std::cout << "Invalid Base Input Argument\n";
    Usage();
  } else if (arg == "-h" || arg == "--help" || arg == "-?")
    Usage();
  else
    return arg;
}
#if 0
Fin::Fin(const int argc, const char* argv[]) 
{
    auto base_arg = ParseBaseArg(argc, argv);
    AddCmdLineArgs();
    ParseCmdLineArgs(argc, argv);
    GetAndSetData();
    AllocateBuffersAndCopy();
    data_type = miopenFloat;
#if MIOPEN_BACKEND_OPENCL
    // miopenCreate(&handle);
    // Default constructor would do
#elif MIOPEN_BACKEND_HIP
    hipStream_t s;
    hipStreamCreate(&s);
    handle = miopen::Handle{s};
#endif

    q = handle->GetStream();
}
#endif
} // namespace fin
