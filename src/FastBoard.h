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

#ifndef FASTBOARD_H_INCLUDED
#define FASTBOARD_H_INCLUDED

#include "config.h"

#include <array>
#include <string>
#include <vector>
#include <queue>

class FastBoard {
    friend class FastState;
public:
    /*
        neighbor counts are up to 4, so 3 bits is ok,
        but a power of 2 makes things a bit faster
    */
    static constexpr int NBR_SHIFT = 4;

    /*
        largest board supported
    */
    static constexpr int MAXBOARDSIZE = 19;

    /*
        highest existing square
    */
    static constexpr int MAXSQ = ((MAXBOARDSIZE + 2) * (MAXBOARDSIZE + 2));

    /*
        infinite score
    */
    static constexpr int BIG = 10000000;

    /*
        vertex of a pass
    */
    static constexpr int PASS   = -1;
    /*
        vertex of a "resign move"
    */
    static constexpr int RESIGN = -2;

    /*
        possible contents of a square
    */
    enum square_t : char {
        BLACK = 0, WHITE = 1, EMPTY = 2, INVAL = 3
    };

    /*
        move generation types
    */
    using movescore_t = std::pair<int, float>;
    using scoredmoves_t = std::vector<movescore_t>;

    int get_boardsize(void) const;
    square_t get_square(int x, int y) const;
    square_t get_square(int vertex) const ;
    int get_vertex(int i, int j) const;
    void set_square(int x, int y, square_t content);
    void set_square(int vertex, square_t content);
    int rotate_vertex(int vertex, int symmetry);
    std::pair<int, int> get_xy(int vertex) const;
    int get_groupid(int vertex);

    bool is_suicide(int i, int color);
    int fast_ss_suicide(const int color, const int i);
    int update_board_fast(const int color, const int i, bool & capture);
    int count_pliberties(const int i);
    int count_rliberties(const int i);
    int merged_string_size(int color, int vertex);
    void augment_chain(std::vector<int> & chains, int vertex);
    bool is_eye(const int color, const int vtx);
    int get_dir(int i);
    int get_extra_dir(int i);

    int estimate_mc_score(float komi);
    float final_mc_score(float komi);
    int get_stone_count();
    float area_score(float komi);
    std::vector<bool> calc_reach_color(int col);

    int get_prisoners(int side);
    bool black_to_move();
    int get_to_move();
    void set_to_move(int color);

    std::string move_to_text(int move);
    std::string move_to_text_sgf(int move);
    int text_to_move(std::string move);
    std::string get_stone_list();
    int string_size(int vertex);
    std::vector<int> get_string_stones(int vertex);
    std::string get_string(int vertex);

    void reset_board(int size);
    void display_liberties(int lastmove = -1);
    void display_board(int lastmove = -1);

    static bool starpoint(int size, int point);
    static bool starpoint(int size, int x, int y);

protected:
    /*
        bit masks to detect eyes on neighbors
    */
    static const std::array<int,      2> s_eyemask;
    static const std::array<square_t, 4> s_cinvert; /* color inversion */

    std::array<square_t, MAXSQ>            m_square;      /* board contents */
    std::array<unsigned short, MAXSQ+1>    m_next;        /* next stone in string */
    std::array<unsigned short, MAXSQ+1>    m_parent;      /* parent node of string */
	/*
	     Liberties per string parent. Liberties for the string that contains the specified vertex.
	     It's the number of liberties for the string of which the index is the "parent" vertex,
	     and only guaranteed to be correct for that parent vertex.
	 */
    std::array<unsigned short, MAXSQ+1>    m_libs;        
    std::array<unsigned short, MAXSQ+1>    m_stones;      /* stones per string parent */
    std::array<unsigned short, MAXSQ>      m_neighbours;  /* counts of neighboring stones */
    std::array<int, 4>                     m_dirs;        /* movement directions 4 way */
    std::array<int, 8>                     m_extradirs;   /* movement directions 8 way */
    std::array<int, 2>                     m_prisoners;   /* prisoners per color */
    std::array<int, 2>                     m_totalstones; /* stones per color */
    std::vector<int>                       m_critical;    /* queue of critical points */
    std::array<unsigned short, MAXSQ>      m_empty;       /* empty squares */
    std::array<unsigned short, MAXSQ>      m_empty_idx;   /* indexes of square */
    int m_empty_cnt;                                      /* count of empties */

    int m_tomove;
    int m_maxsq;

    int m_boardsize;

    int count_neighbours(const int color, const int i);
    void merge_strings(const int ip, const int aip);
    int remove_string_fast(int i);
    void add_neighbour(const int i, const int color);
    void remove_neighbour(const int i, const int color);
    int update_board_eye(const int color, const int i);
    int in_atari(int vertex);
    bool fast_in_atari(int vertex);
	void print_column_labels(int size, std::string padding = " ");
};

#endif
