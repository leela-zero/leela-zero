/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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
*/

#ifndef SGFTREE_H_INCLUDED
#define SGFTREE_H_INCLUDED

#include <stddef.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "FastBoard.h"
#include "GameState.h"
#include "KoState.h"

class SGFTree {
public:
    static const int EOT = 0;               // End-Of-Tree marker

    SGFTree() = default;
    void init_state();

    KoState * get_state();
    GameState follow_mainline_state(unsigned int movenum = 999);
    std::vector<int> get_mainline();
    void load_from_file(std::string filename, int index = 0);
    void load_from_string(std::string gamebuff);

    void add_property(std::string property, std::string value);
    SGFTree * add_child();
    SGFTree * get_child(size_t count);
    int get_move(int tomove);
    bool is_initialized() const {
        return m_initialized;
    }
    FastBoard::square_t get_winner() const;

    static std::string state_to_string(GameState& state, int compcolor);

private:
    void populate_states(void);
    void apply_move(int color, int move);
    void apply_move(int move);
    void copy_state(const SGFTree& state);
    int string_to_vertex(const std::string& move) const;

    using PropertyMap = std::multimap<std::string, std::string>;

    bool m_initialized{false};
    KoState m_state;
    FastBoard::square_t m_winner{FastBoard::INVAL};
    std::vector<SGFTree> m_children;
    PropertyMap m_properties;
};

#endif
