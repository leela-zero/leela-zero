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

#ifndef SGFTREE_H_INCLUDED
#define SGFTREE_H_INCLUDED

#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "FastBoard.h"
#include "GameState.h"
#include "KoState.h"

class SGFTree {
public:
    static constexpr auto EOT = 0;               // End-Of-Tree marker

    SGFTree() = default;
    void init_state();

    const KoState * get_state() const;
    GameState follow_mainline_state(unsigned int movenum = 999) const;
    std::vector<int> get_mainline() const;

    void load_from_file(const std::string& filename, int index = 0);
    void load_from_string(const std::string& gamebuff);

    void add_property(std::string property, std::string value);
    SGFTree * add_child();
    const SGFTree * get_child(size_t count) const;
    int get_move(int tomove) const;
    std::pair<int, int> get_colored_move() const;
    bool is_initialized() const {
        return m_initialized;
    }
    FastBoard::vertex_t get_winner() const;

    static std::string state_to_string(GameState& state, int compcolor);

private:
    void populate_states();
    void apply_move(int color, int move);
    void apply_move(int move);
    void copy_state(const SGFTree& state);
    int string_to_vertex(const std::string& move) const;

    using PropertyMap = std::multimap<std::string, std::string>;

    bool m_initialized{false};
    KoState m_state;
    FastBoard::vertex_t m_winner{FastBoard::INVAL};
    std::vector<SGFTree> m_children;
    PropertyMap m_properties;
};

#endif
