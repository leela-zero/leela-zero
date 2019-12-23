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

#include "config.h"
#include "SGFTree.h"

#include <cassert>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <ctime>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "FullBoard.h"
#include "GTP.h"
#include "KoState.h"
#include "SGFParser.h"
#include "Utils.h"

using namespace Utils;

const int SGFTree::EOT;

void SGFTree::init_state() {
    m_initialized = true;
    // Initialize with defaults.
    // The SGF might be missing boardsize or komi
    // which means we'll never initialize properly.
    m_state.init_game(std::min(BOARD_SIZE, 19), TRAINED_UNIT_KOMI);
}

const KoState * SGFTree::get_state(void) const {
    assert(m_initialized);
    return &m_state;
}

const SGFTree * SGFTree::get_child(size_t count) const {
    if (count < m_children.size()) {
        assert(m_initialized);
        return &(m_children[count]);
    } else {
        return nullptr;
    }
}

// This follows the entire line, and doesn't really need the intermediate
// states, just the moves. As a consequence, states that contain more than
// just moves won't have any effect.
GameState SGFTree::follow_mainline_state(unsigned int movenum) const {
    const auto* link = this;
    // This initializes a starting state from a KoState and
    // sets up the game history.
    GameState result(get_state());

    if (m_timecontrol_ptr) {
        result.set_timecontrol(*m_timecontrol_ptr);
    }

    for (unsigned int i = 0; i <= movenum && link != nullptr; i++) {
        // root position has no associated move
        if (i != 0) {
            auto colored_move = link->get_colored_move();
            if (colored_move.first != FastBoard::INVAL) {
                if (colored_move.second != FastBoard::PASS
                    && colored_move.second != FastBoard::EMPTY
                    && result.board.get_state(colored_move.second)
                       != FastBoard::EMPTY) {
                    // Fail loading
                    return result;
                }
                result.play_move(colored_move.first, colored_move.second);
            }
        }
        link = link->get_child(0);
    }

    return result;
}

void SGFTree::load_from_string(const std::string& gamebuff) {
    std::istringstream pstream(gamebuff);

    // loads properties with moves
    SGFParser::parse(pstream, this);

    // Set up the root state to defaults
    init_state();

    // populates the states from the moves
    // split this up in root node, achor (handicap), other nodes
    populate_states();
}

// load a single game from a file
void SGFTree::load_from_file(const std::string& filename, int index) {
    auto gamebuff = SGFParser::chop_from_file(filename, index);

    //myprintf("Parsing: %s\n", gamebuff.c_str());

    load_from_string(gamebuff);
}

void SGFTree::populate_states() {
    PropertyMap::iterator it;
    auto valid_size = false;
    auto has_handicap = false;

    // first check for go game setup in properties
    it = m_properties.find("GM");
    if (it != end(m_properties)) {
        if (it->second != "1") {
            throw std::runtime_error("SGF Game is not a Go game");
        } else {
            if (!m_properties.count("SZ")) {
                // No size, but SGF spec defines default size for Go
                m_properties.insert(std::make_pair("SZ", "19"));
                valid_size = true;
            }
        }
    }

    // board size
    it = m_properties.find("SZ");
    if (it != end(m_properties)) {
        const auto size = it->second;
        std::istringstream strm(size);
        int bsize;
        strm >> bsize;
        if (bsize == BOARD_SIZE) {
            // Assume default komi in config.h if not specified
            m_state.init_game(bsize, TRAINED_UNIT_KOMI);
            valid_size = true;
        } else {
            throw std::runtime_error("Board size not supported.");
        }
    }

    // komi
    it = m_properties.find("KM");
    if (it != end(m_properties)) {
        const auto foo = it->second;
        std::istringstream strm(foo);
        float komi;
        strm >> komi;
        const auto handicap = m_state.get_handicap();
        // last ditch effort: if no GM or SZ, assume 19x19 Go here
        auto bsize = 19;
        if (valid_size) {
            bsize = m_state.board.get_boardsize();
        }
        if (bsize == BOARD_SIZE) {
            m_state.init_game(bsize, komi);
            m_state.set_handicap(handicap);
        } else {
            throw std::runtime_error("Board size not supported.");
        }
    }

    // time
    it = m_properties.find("TM");
    if (it != end(m_properties)) {
        const auto maintime = it->second;
        it = m_properties.find("OT");
        const auto byoyomi = (it != end(m_properties)) ? it->second : "";
        it = m_properties.find("BL");
        const auto black_time_left = (it != end(m_properties)) ? it->second : "";
        it = m_properties.find("WL");
        const auto white_time_left = (it != end(m_properties)) ? it->second : "";
        it = m_properties.find("OB");
        const auto black_moves_left = (it != end(m_properties)) ? it->second : "";
        it = m_properties.find("OW");
        const auto white_moves_left = (it != end(m_properties)) ? it->second : "";
        m_timecontrol_ptr = TimeControl::make_from_text_sgf(maintime, byoyomi,
                                                            black_time_left,
                                                            white_time_left,
                                                            black_moves_left,
                                                            white_moves_left);
    }

    // handicap
    it = m_properties.find("HA");
    if (it != end(m_properties)) {
        const auto size = it->second;
        std::istringstream strm(size);
        float handicap;
        strm >> handicap;
        has_handicap = (handicap > 0.0f);
        m_state.set_handicap(int(handicap));
    }

    // result
    it = m_properties.find("RE");
    if (it != end(m_properties)) {
        const auto result = it->second;
        if (boost::algorithm::find_first(result, "Time")) {
            // std::cerr << "Skipping: " << result << std::endl;
            m_winner = FastBoard::EMPTY;
        } else {
            if (boost::algorithm::starts_with(result, "W+")) {
                m_winner = FastBoard::WHITE;
            } else if (boost::algorithm::starts_with(result, "B+")) {
                m_winner = FastBoard::BLACK;
            } else {
                m_winner = FastBoard::INVAL;
                // std::cerr << "Could not parse game result: " << result << std::endl;
            }
        }
    } else {
        m_winner = FastBoard::EMPTY;
    }

    // handicap stones
    auto prop_pair_ab = m_properties.equal_range("AB");
    // Do we have a handicap specified but no handicap stones placed in
    // the same node? Then the SGF file is corrupt. Let's see if we can find
    // them in the next node, which is a common bug in some Go apps.
    if (has_handicap && prop_pair_ab.first == prop_pair_ab.second) {
        if (!m_children.empty()) {
            auto& successor = m_children[0];
            prop_pair_ab = successor.m_properties.equal_range("AB");
        }
    }
    // Loop through the stone list and apply
    for (auto pit = prop_pair_ab.first; pit != prop_pair_ab.second; ++pit) {
        const auto move = pit->second;
        const auto vtx = string_to_vertex(move);
        apply_move(FastBoard::BLACK, vtx);
    }

    // XXX: count handicap stones
    const auto& prop_pair_aw = m_properties.equal_range("AW");
    for (auto pit = prop_pair_aw.first; pit != prop_pair_aw.second; ++pit) {
        const auto move = pit->second;
        const auto vtx = string_to_vertex(move);
        apply_move(FastBoard::WHITE, vtx);
    }

    it = m_properties.find("PL");
    if (it != end(m_properties)) {
        const auto who = it->second;
        if (who == "W") {
            m_state.set_to_move(FastBoard::WHITE);
        } else if (who == "B") {
            m_state.set_to_move(FastBoard::BLACK);
        }
    }

    // now for all children play out the moves
    for (auto& child_state : m_children) {
        // propagate state
        child_state.copy_state(*this);

        // XXX: maybe move this to the recursive call
        // get move for side to move
        const auto colored_move = child_state.get_colored_move();
        if (colored_move.first != FastBoard::INVAL) {
            child_state.apply_move(colored_move.first, colored_move.second);
        }

        child_state.populate_states();
    }
}

void SGFTree::copy_state(const SGFTree& tree) {
    m_initialized = tree.m_initialized;
    m_state = tree.m_state;
    m_timecontrol_ptr = tree.m_timecontrol_ptr;
}

void SGFTree::apply_move(int color, int move) {
    if (move != FastBoard::PASS && move != FastBoard::RESIGN) {
        auto vtx_state = m_state.board.get_state(move);
        if (vtx_state == !color || vtx_state == FastBoard::INVAL) {
            throw std::runtime_error("Illegal move");
        }
        // Playing on an occupied intersection is legal in SGF setup,
        // but we can't really handle it. So just ignore and hope that works.
        if (vtx_state == color) {
            return;
        }
        assert(vtx_state == FastBoard::EMPTY);
    }
    m_state.play_move(color, move);
}

void SGFTree::apply_move(int move) {
    auto color = m_state.get_to_move();
    apply_move(color, move);
}

void SGFTree::add_property(std::string property, std::string value) {
    m_properties.emplace(property, value);
}

SGFTree * SGFTree::add_child() {
    // first allocation is better small
    if (m_children.size() == 0) {
        m_children.reserve(1);
    }
    m_children.emplace_back();
    return &(m_children.back());
}

int SGFTree::string_to_vertex(const std::string& movestring) const {
    if (movestring.size() == 0) {
        return FastBoard::PASS;
    }

    if (m_state.board.get_boardsize() <= 19) {
        if (movestring == "tt") {
            return FastBoard::PASS;
        }
    }

    int bsize = m_state.board.get_boardsize();
    if (bsize == 0) {
        throw std::runtime_error("Node has 0 sized board");
    }

    char c1 = movestring[0];
    char c2 = movestring[1];

    int cc1;
    int cc2;

    if (c1 >= 'A' && c1 <= 'Z') {
        cc1 = 26 + c1 - 'A';
    } else {
        cc1 = c1 - 'a';
    }
    if (c2 >= 'A' && c2 <= 'Z') {
        cc2 = bsize - 26 - (c2 - 'A') - 1;
    } else {
        cc2 = bsize - (c2 - 'a') - 1;
    }

    // catch illegal SGF
    if (cc1 < 0 || cc1 >= bsize
        || cc2 < 0 || cc2 >= bsize) {
        throw std::runtime_error("Illegal SGF move");
    }

    int vtx = m_state.board.get_vertex(cc1, cc2);

    return vtx;
}

int SGFTree::get_move(int tomove) const {
    std::string colorstring;

    if (tomove == FastBoard::BLACK) {
        colorstring = "B";
    } else {
        colorstring = "W";
    }

    auto it = m_properties.find(colorstring);
    if (it != end(m_properties)) {
        std::string movestring = it->second;
        return string_to_vertex(movestring);
    }

    return SGFTree::EOT;
}

std::pair<int, int> SGFTree::get_colored_move() const {
    for (const auto& prop : m_properties) {
        if (prop.first == "B") {
            return std::make_pair(FastBoard::BLACK,
                                  string_to_vertex(prop.second));
        } else if (prop.first == "W") {
            return std::make_pair(FastBoard::WHITE,
                                  string_to_vertex(prop.second));
        }
    }
    return std::make_pair(FastBoard::INVAL, SGFTree::EOT);
}

FastBoard::vertex_t SGFTree::get_winner() const {
    return m_winner;
}

std::vector<int> SGFTree::get_mainline() const {
    std::vector<int> moves;

    const auto* link = this;
    auto tomove = link->m_state.get_to_move();
    link = link->get_child(0);

    while (link != nullptr && link->is_initialized()) {
        auto move = link->get_move(tomove);
        if (move != SGFTree::EOT) {
            moves.push_back(move);
        }
        tomove = !tomove;
        link = link->get_child(0);
    }

    return moves;
}

std::string SGFTree::state_to_string(GameState& pstate, int compcolor) {
    auto state = std::make_unique<GameState>();

    // make a working copy
    *state = pstate;

    std::string header;
    std::string moves;

    auto komi = state->get_komi();
    auto size = state->board.get_boardsize();
    time_t now;
    time(&now);
    char timestr[sizeof "2017-10-16"];
    strftime(timestr, sizeof timestr, "%F", localtime(&now));

    header.append("(;GM[1]FF[4]RU[Chinese]");
    header.append("DT[" + std::string(timestr) + "]");
    header.append("SZ[" + std::to_string(size) + "]");
    header.append("KM[" + str(boost::format("%.1f") % komi) + "]");
    header.append(state->get_timecontrol().to_text_sgf());

    auto leela_name = std::string{PROGRAM_NAME};
    leela_name.append(" " + std::string(PROGRAM_VERSION));
    if (!cfg_weightsfile.empty()) {
        auto pos = cfg_weightsfile.find_last_of("\\/");
        if (std::string::npos == pos) {
            pos = 0;
        } else {
            ++pos;
        }
        leela_name.append(" " + cfg_weightsfile.substr(pos, 8));
    }

    if (compcolor == FastBoard::WHITE) {
        header.append("PW[" + leela_name + "]");
        header.append("PB[Human]");
    } else {
        header.append("PB[" + leela_name + "]");
        header.append("PW[Human]");
    }

    state->rewind();

    // check handicap here (anchor point)
    auto handicap = 0;
    std::string handicapstr;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int vertex = state->board.get_vertex(i, j);
            int vtx_state = state->board.get_state(vertex);

            if (vtx_state == FastBoard::BLACK) {
                handicap++;
                handicapstr.append("[" + state->board.move_to_text_sgf(vertex) + "]");
            }
        }
    }

    if (handicap > 0) {
        header.append("HA[" + std::to_string(handicap) + "]");
        moves.append("AB" + handicapstr);
    }

    moves.append("\n");

    int counter = 0;

    while (state->forward_move()) {
        int move = state->get_last_move();
        assert(move != FastBoard::RESIGN);
        std::string movestr = state->board.move_to_text_sgf(move);
        if (state->board.black_to_move()) {
            moves.append(";W[" + movestr + "]");
        } else {
            moves.append(";B[" + movestr + "]");
        }
        if (++counter % 10 == 0) {
            moves.append("\n");
        }
    }

    if (!state->has_resigned()) {
        float score = state->final_score();

        if (score > 0.0f) {
            header.append("RE[B+" + str(boost::format("%.1f") % score) + "]");
        } else if (score < 0.0f) {
            header.append("RE[W+" + str(boost::format("%.1f") % -score) + "]");
        } else {
            header.append("RE[0]");
        }
    } else {
        if (state->who_resigned() == FastBoard::WHITE) {
            header.append("RE[B+Resign]");
        } else {
            header.append("RE[W+Resign]");
        }
    }

    header.append("\nC[" + std::string{PROGRAM_NAME} + " options:" + cfg_options_str + "]");

    std::string result(header);
    result.append("\n");
    result.append(moves);
    result.append(")\n");

    return result;
}
