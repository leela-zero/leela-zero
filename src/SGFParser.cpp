/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "SGFParser.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

#include "SGFTree.h"
#include "Utils.h"

std::vector<SGFParser::ELF_Data> SGFParser::chop_elf(std::string filename,
                                                     size_t stopat) {
    std::ifstream ins(filename.c_str(), std::ifstream::binary | std::ifstream::in);

    if (ins.fail()) {
        throw std::runtime_error("Error opening file");
    }

    std::vector<ELF_Data> result;
    std::string gamebuff;

    auto constexpr MAX = std::numeric_limits<std::streamsize>::max();
    std::string str;
    char c;
    int num_move;
    int total_policy;

    while (result.size() <= stopat) {
        while (true) {
            ins.ignore(MAX, '\"');
            if ((ins.rdstate() & std::ifstream::eofbit) != 0) return result;
            std::getline(ins, str, ':');
            if (str == "content\"") break;
        }
        result.emplace_back(ELF_Data());
        auto& data = result.back();
        ins >> c; // eat "
        str.clear();
        std::getline(ins, str, '\"');
        data.sgf = str;
        ins.ignore(MAX, ':'); // eat ,"num_move":
        str.clear();
        std::getline(ins, str, ',');
        num_move = std::stoi(str);
        ins.ignore(MAX, '['); // eat "policies":[
        data.policies.clear();
        data.policies.reserve(num_move + 1);
        while (true) {
            ins >> c; // eat [ (assume there's at least one policy record)
            str.clear();
            std::getline(ins, str, ',');
            data.policies.emplace_back(std::array<float, POTENTIAL_MOVES>());
            auto& probs = data.policies.back();
            probs[NUM_INTERSECTIONS] = total_policy = std::stoi(str);
            for (auto i = 0; i < BOARD_SIZE; i++) ins.ignore(MAX, ',');
            for (int i = BOARD_SIZE - 1; i >= 0; i--) {
                ins.ignore(MAX, ',');
                ins.ignore(MAX, ',');
                for (auto j = 0; j < BOARD_SIZE; j++) {
                    str.clear();
                    std::getline(ins, str, ',');
                    auto entry = std::stoi(str);
                    total_policy += entry;
                    probs[i * BOARD_SIZE + j] = entry;
                }
            }
            for (auto& p : probs) p /= total_policy;
            ins.ignore(MAX, ']');
            ins >> c;
            if (c == ']') break;
        }
        ins.ignore(MAX, ':'); // eat ,"reward":
        str.clear();
        std::getline(ins, str, ',');
        data.reward = std::stof(str);
        auto move_excess = (str.back() == '0') ? 1 : 0;
        auto train_pos = data.policies.size();
        if (train_pos != num_move + move_excess) {
            Utils::myprintf("Spurious ELF data entry! ");
            Utils::myprintf("Moves: %d, Policy records: %d, Reward: %s\n",
                             num_move, train_pos, str.c_str(), str.back());
        }
    }
}

std::vector<std::string> SGFParser::chop_stream(std::istream& ins,
                                                size_t stopat) {
    std::vector<std::string> result;
    std::string gamebuff;

    ins >> std::noskipws;

    int nesting = 0;      // parentheses
    bool intag = false;   // brackets
    int line = 0;
    gamebuff.clear();

    char c;
    while (ins >> c && result.size() <= stopat) {
        if (c == '\n') line++;

        gamebuff.push_back(c);
        if (c == '\\') {
            // read literal char
            ins >> c;
            gamebuff.push_back(c);
            // Skip special char parsing
            continue;
        }

        if (c == '(' && !intag) {
            if (nesting == 0) {
                // eat ; too
                do {
                    ins >> c;
                } while (std::isspace(c) && c != ';');
                gamebuff.clear();
            }
            nesting++;
        } else if (c == ')' && !intag) {
            nesting--;

            if (nesting == 0) {
                result.push_back(gamebuff);
            }
        } else if (c == '[' && !intag) {
            intag = true;
        } else if (c == ']') {
            if (intag == false) {
                Utils::myprintf("Tag error on line %d", line);
            }
            intag = false;
        }
    }

    // No game found? Assume closing tag was missing (OGS)
    if (result.size() == 0) {
        result.push_back(gamebuff);
    }

    return result;
}

std::vector<std::string> SGFParser::chop_all(std::string filename,
                                             size_t stopat) {
    std::ifstream ins(filename.c_str(), std::ifstream::binary | std::ifstream::in);

    if (ins.fail()) {
        throw std::runtime_error("Error opening file");
    }

    auto result = chop_stream(ins, stopat);
    ins.close();

    return result;
}

// scan the file and extract the game with number index
std::string SGFParser::chop_from_file(std::string filename, size_t index) {
    auto vec = chop_all(filename, index);
    return vec[index];
}

std::string SGFParser::parse_property_name(std::istringstream & strm) {
    std::string result;

    char c;
    while (strm >> c) {
        // SGF property names are guaranteed to be uppercase,
        // except that some implementations like IGS are retarded
        // and don't folow the spec. So allow both upper/lowercase.
        if (!std::isupper(c) && !std::islower(c)) {
            strm.unget();
            break;
        } else {
            result.push_back(c);
        }
    }

    return result;
}

bool SGFParser::parse_property_value(std::istringstream & strm,
                                     std::string & result) {
    strm >> std::noskipws;

    char c;
    while (strm >> c) {
        if (!std::isspace(c)) {
            strm.unget();
            break;
        }
    }

    strm >> c;

    if (c != '[') {
        strm.unget();
        return false;
    }

    while (strm >> c) {
        if (c == ']') {
            break;
        } else if (c == '\\') {
            strm >> c;
        }
        result.push_back(c);
    }

    strm >> std::skipws;

    return true;
}

void SGFParser::parse(std::istringstream & strm, SGFTree * node) {
    bool splitpoint = false;

    char c;
    while (strm >> c) {
        if (strm.fail()) {
            return;
        }

        if (std::isspace(c)) {
            continue;
        }

        // parse a property
        if (std::isalpha(c) && std::isupper(c)) {
            strm.unget();

            std::string propname = parse_property_name(strm);
            bool success;

            do {
                std::string propval;
                success = parse_property_value(strm, propval);
                if (success) {
                    node->add_property(propname, propval);
                }
            } while (success);

            continue;
        }

        if (c == '(') {
            // eat first ;
            char cc;
            do {
                strm >> cc;
            } while (std::isspace(cc));
            if (cc != ';') {
                strm.unget();
            }
            // start a variation here
            splitpoint = true;
            // new node
            SGFTree * newptr = node->add_child();
            parse(strm, newptr);
        } else if (c == ')') {
            // variation ends, go back
            // if the variation didn't start here, then
            // push the "variation ends" mark back
            // and try again one level up the tree
            if (!splitpoint) {
                strm.unget();
                return;
            } else {
                splitpoint = false;
                continue;
            }
        } else if (c == ';') {
            // new node
            SGFTree * newptr = node->add_child();
            node = newptr;
            continue;
        }
    }
}
