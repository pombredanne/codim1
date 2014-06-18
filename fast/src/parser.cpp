#include "parser.h"

using namespace codim1;

YAML::Node codim1::load_file(std::string filename) {
    YAML::Node file = YAML::LoadFile(filename);
    return file;
}
