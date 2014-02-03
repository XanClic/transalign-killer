#ifndef _TRANSALIGN_KILLER_THRUST_H_
#define _TRANSALIGN_KILLER_THRUST_H_
#include <iostream>
#include <fstream>
#include <string>

void print_help(){

  std::cout << "usage: \t transalign_killer <file/to/read/sequence/from>\n";
}


std::string get_file_contents(const char *filename)
{
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  std::string contents("");
  // print file:
  if(in.is_open()){

    while (in.good()) {
      contents.push_back(in.get());
    }
  }
  else
    {
      std::cerr << ">> problem opening file at: " << filename << "\n";
    }

  return contents;

}

#endif /* _TRANSALIGN_KILLER_H_ */
