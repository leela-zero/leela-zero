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

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <string>
#include <sstream>
#include <cmath>
#include <climits>
#include <algorithm>

#include "config.h"
#include "Utils.h"
#include "GameState.h"
#include "GTP.h"
#include "UCTSearch.h"
#include "UCTNode.h"
#include "SGFTree.h"
#include "Network.h"
#include "TTable.h"
#include "Training.h"

using namespace Utils;

// Configuration flags
bool cfg_allow_pondering;
int cfg_tcp_port;
int cfg_num_threads;
int cfg_max_playouts;
int cfg_lagbuffer_cs;
int cfg_resignpct;
int cfg_noise;
int cfg_random_cnt;
bool cfg_dumbpass;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
int cfg_rowtiles;
#endif
float cfg_puct;
float cfg_softmax_temp;
std::string cfg_weightsfile;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;
extern unsigned char * mem;
extern int myid;

void GTP::setup_default_parameters() {
    cfg_allow_pondering = true;
    cfg_tcp_port = 9999;
    cfg_num_threads = std::max(1, std::min(SMP::get_num_cpus(), MAX_CPUS));
    cfg_max_playouts = std::numeric_limits<decltype(cfg_max_playouts)>::max();
    cfg_lagbuffer_cs = 100;
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_rowtiles = 5;
#endif
    cfg_puct = 0.85f;
    cfg_softmax_temp = 1.0f;
    cfg_resignpct = 10;
    cfg_noise = false;
    cfg_random_cnt = 0;
    cfg_dumbpass = false;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
}

const std::string GTP::s_commands[] = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "undo",
    "final_score",
    "final_status_list",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "place_free_handicap",
    "set_free_handicap",
    "loadsgf",
    "printsgf",
    "kgs-genmove_cleanup",
    "kgs-time_settings",
    "kgs-game_over",
    "heatmap",
    ""
};

std::string GTP::get_life_list(GameState & game, bool live) {
    std::vector<std::string> stringlist;
    std::string result;
    FastBoard & board = game.board;

    if (live) {
        for (int i = 0; i < board.get_boardsize(); i++) {
            for (int j = 0; j < board.get_boardsize(); j++) {
                int vertex = board.get_vertex(i, j);

                if (board.get_square(vertex) != FastBoard::EMPTY) {
                    stringlist.push_back(board.get_string(vertex));
                }
            }
        }
    }

    // remove multiple mentions of the same string
    // unique reorders and returns new iterator, erase actually deletes
    std::sort(stringlist.begin(), stringlist.end());
    stringlist.erase(std::unique(stringlist.begin(), stringlist.end()),
                     stringlist.end());

    for (size_t i = 0; i < stringlist.size(); i++) {
        result += (i == 0 ? "" : "\n") + stringlist[i];
    }

    return result;
}

bool GTP::execute(GameState & game, std::string xinput) {
    std::string input;

    bool transform_lowercase = true;

    // Required on Unixy systems
    if (xinput.find("loadsgf") != std::string::npos) {
        transform_lowercase = false;
    }

    /* eat empty lines, simple preprocessing, lower case */
    for (unsigned int tmp = 0; tmp < xinput.size(); tmp++) {
        if (xinput[tmp] == 9) {
            input += " ";
        } else if ((xinput[tmp] > 0 && xinput[tmp] <= 9)
	        || (xinput[tmp] >= 11 && xinput[tmp] <= 31)
	        || xinput[tmp] == 127) {
	       continue;
        } else {
            if (transform_lowercase) {
                input += std::tolower(xinput[tmp]);
            } else {
                input += xinput[tmp];
            }
        }

        // eat multi whitespace
        if (input.size() > 1) {
            if (std::isspace(input[input.size() - 2]) &&
                std::isspace(input[input.size() - 1])) {
                input.resize(input.size() - 1);
            }
        }
    }

    std::string command;
    int id = -1;

    if (input == "") {
        return true;
    } else if (input == "exit") {
        exit(EXIT_SUCCESS);
        return true;
    } else if (input == "#") {
        return true;
    } else if (std::isdigit(input[0])) {
        std::istringstream strm(input);
        char spacer;
        strm >> id;
        strm >> std::noskipws >> spacer;
        std::getline(strm, command);
    } else {
        command = input;
    }

    /* process commands */
    if (command == "protocol_version") {
        gtp_printf(id, "%d", GTP_VERSION);
        return true;
    } else if (command == "name") {
        gtp_printf(id, PROGRAM_NAME);
        return true;
    } else if (command == "version") {
        gtp_printf(id, PROGRAM_VERSION);
        return true;
    } else if (command == "quit") {
        // free the slot
        mem[2+myid] = 0;
        gtp_printf(id, "");
        exit(EXIT_SUCCESS);
    } else if (command.find("known_command") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;     /* remove known_command */
        cmdstream >> tmp;

        for (int i = 0; s_commands[i].size() > 0; i++) {
            if (tmp == s_commands[i]) {
                gtp_printf(id, "true");
                return 1;
            }
        }

        gtp_printf(id, "false");
        return true;
    } else if (command.find("list_commands") == 0) {
        std::string outtmp(s_commands[0]);
        for (int i = 1; s_commands[i].size() > 0; i++) {
            outtmp = outtmp + "\n" + s_commands[i];
        }
        gtp_printf(id, outtmp.c_str());
        return true;
    } else if (command.find("boardsize") == 0) {
        std::istringstream cmdstream(command);
        std::string stmp;
        int tmp;

        cmdstream >> stmp;  // eat boardsize
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            if (tmp != 19) {
                gtp_fail_printf(id, "unacceptable size");
            } else {
                float old_komi = game.get_komi();
                Training::clear_training();
                game.init_game(tmp, old_komi);
                gtp_printf(id, "");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("clear_board") == 0) {
        Training::clear_training();
        game.reset_game();
        gtp_printf(id, "");
        return true;
    } else if (command.find("komi") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        float komi = 7.5f;
        float old_komi = game.get_komi();

        cmdstream >> tmp;  // eat komi
        cmdstream >> komi;

        if (!cmdstream.fail()) {
            if (komi != old_komi) {
                game.set_komi(komi);
            }
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("play") == 0) {
        if (command.find("pass") != std::string::npos
            || command.find("resign") != std::string::npos) {
            game.play_pass();
            gtp_printf(id, "");
        } else {
            std::istringstream cmdstream(command);
            std::string tmp;
            std::string color, vertex;

            cmdstream >> tmp;   //eat play
            cmdstream >> color;
            cmdstream >> vertex;

            if (!cmdstream.fail()) {
                if (!game.play_textmove(color, vertex)) {
                    gtp_fail_printf(id, "illegal move");
                } else {
                    gtp_printf(id, "");
                }
            } else {
                gtp_fail_printf(id, "syntax not understood");
            }
        }
        return true;
    } else if (command.find("genmove") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat genmove
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return 1;
            }
            // start thinking
            {
                auto search = std::make_unique<UCTSearch>(game);

                int move = search->think(who);
                game.play_move(who, move);

                std::string vertex = game.move_to_text(move);
                gtp_printf(id, "%s", vertex.c_str());
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (game.get_last_move() != FastBoard::RESIGN) {
                    auto search = std::make_unique<UCTSearch>(game);
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("kgs-genmove_cleanup") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat kgs-genmove
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return 1;
            }
            game.set_passes(0);
            {
                auto search = std::make_unique<UCTSearch>(game);

                int move = search->think(who, UCTSearch::NOPASS);
                game.play_move(who, move);

                std::string vertex = game.move_to_text(move);
                gtp_printf(id, "%s", vertex.c_str());
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (game.get_last_move() != FastBoard::RESIGN) {
                    auto search = std::make_unique<UCTSearch>(game);
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("undo") == 0) {
        if (game.undo_move()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "cannot undo");
        }
        return true;
    } else if (command.find("showboard") == 0) {
        gtp_printf(id, "");
        game.display_state();
        return true;
    } else if (command.find("mc_score") == 0) {
        float ftmp = game.board.final_mc_score(game.get_komi());
        /* white wins */
        if (ftmp < -0.1) {
            gtp_printf(id, "W+%3.1f", (float)fabs(ftmp));
        } else if (ftmp > 0.1) {
            gtp_printf(id, "B+%3.1f", ftmp);
        } else {
            gtp_printf(id, "0");
        }
        return true;
    } else if (command.find("final_score") == 0) {
        float ftmp = game.final_score();
        /* white wins */
        if (ftmp < -0.1) {
            gtp_printf(id, "W+%3.1f", (float)fabs(ftmp));
        } else if (ftmp > 0.1) {
            gtp_printf(id, "B+%3.1f", ftmp);
        } else {
            gtp_printf(id, "0");
        }
        return true;
    } else if (command.find("final_status_list") == 0) {
        if (command.find("alive") != std::string::npos) {
            std::string livelist = get_life_list(game, true);
            gtp_printf(id, livelist.c_str());
        } else if (command.find("dead") != std::string::npos) {
            std::string deadlist = get_life_list(game, false);
            gtp_printf(id, deadlist.c_str());
        } else {
            gtp_printf(id, "");
        }
        return true;
    } else if (command.find("time_settings") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int maintime, byotime, byostones;

        cmdstream >> tmp >> maintime >> byotime >> byostones;

        if (!cmdstream.fail()) {
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);

            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("time_left") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, color;
        int time, stones;

        cmdstream >> tmp >> color >> time >> stones;

        if (!cmdstream.fail()) {
            int icolor;

            if (color == "w" || color == "white") {
                icolor = FastBoard::WHITE;
            } else if (color == "b" || color == "black") {
                icolor = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "Color in time adjust not understood.\n");
                return 1;
            }

            game.adjust_time(icolor, time * 100, stones);

            gtp_printf(id, "");

            if (cfg_allow_pondering) {
                // KGS sends this after our move
                // now start pondering
                if (game.get_last_move() != FastBoard::RESIGN) {
                    auto search = std::make_unique<UCTSearch>(game);
                    search->ponder();
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("auto") == 0) {
        do {
            auto search = std::make_unique<UCTSearch>(game);

            int move = search->think(game.get_to_move(), UCTSearch::NORMAL);
            game.play_move(move);
            game.display_state();

        } while (game.get_passes() < 2
                 && game.get_last_move() != FastBoard::RESIGN);

        return true;
    } else if (command.find("go") == 0) {
        auto search = std::make_unique<UCTSearch>(game);

        int move = search->think(game.get_to_move());
        game.play_move(move);

        std::string vertex = game.move_to_text(move);
        myprintf("%s\n", vertex.c_str());
        return true;
    } else if (command.find("heatmap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int rotation;

        cmdstream >> tmp;   // eat heatmap
        cmdstream >> rotation;

        if (!cmdstream.fail()) {
            auto vec = Network::get_scored_moves(
                &game, Network::Ensemble::DIRECT, rotation);
            Network::show_heatmap(&game, vec, false);
        } else {
            auto vec = Network::get_scored_moves(
                &game, Network::Ensemble::DIRECT, 0);
            Network::show_heatmap(&game, vec, false);
        }
        gtp_printf(id, "");
        return true;
    } else if (command.find("fixed_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat fixed_handicap
        cmdstream >> stones;

        if (game.set_fixed_handicap(stones)) {
            std::string stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }
        return true;
    } else if (command.find("place_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat place_free_handicap
        cmdstream >> stones;

        game.place_free_handicap(stones);

        std::string stonestring = game.board.get_stone_list();
        gtp_printf(id, "%s", stonestring.c_str());

        return true;
    } else if (command.find("set_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;   // eat set_free_handicap

        do {
            std::string vertex;

            cmdstream >> vertex;

            if (!cmdstream.fail()) {
                if (!game.play_textmove("black", vertex)) {
                    gtp_fail_printf(id, "illegal move");
                } else {
                    game.set_handicap(game.get_handicap() + 1);
                }
            }
        } while (!cmdstream.fail());

        std::string stonestring = game.board.get_stone_list();
        gtp_printf(id, "%s", stonestring.c_str());

        return true;
    } else if (command.find("loadsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;
        int movenum;

        cmdstream >> tmp;   // eat loadsgf
        cmdstream >> filename;

        if (!cmdstream.fail()) {
            cmdstream >> movenum;

            if (cmdstream.fail()) {
                movenum = 999;
            }
        } else {
            gtp_fail_printf(id, "Missing filename.");
            return true;
        }

        auto sgftree = std::make_unique<SGFTree>();

        try {
            sgftree->load_from_file(filename);
            game = sgftree->follow_mainline_state(movenum - 1);
            gtp_printf(id, "");
        } catch (const std::exception&) {
            gtp_fail_printf(id, "cannot load file");
        }
        return true;
    } else if (command.find("kgs-chat") == 0) {
        // kgs-chat (game|private) Name Message
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp; // eat kgs-chat
        cmdstream >> tmp; // eat game|private
        cmdstream >> tmp; // eat player name
        do {
            cmdstream >> tmp; // eat message
        } while (!cmdstream.fail());

        gtp_fail_printf(id, "I'm a go bot, not a chat bot.");
        return true;
    } else if (command.find("kgs-game_over") == 0) {
        // Do nothing. Particularly, don't ponder.
        gtp_printf(id, "");
        return true;
    } else if (command.find("kgs-time_settings") == 0) {
        // none, absolute, byoyomi, or canadian
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string tc_type;
        int maintime, byotime, byostones, byoperiods;

        cmdstream >> tmp >> tc_type;

        if (tc_type.find("none") != std::string::npos) {
            // 30 mins
            game.set_timecontrol(30 * 60 * 100, 0, 0, 0);
        } else if (tc_type.find("absolute") != std::string::npos) {
            cmdstream >> maintime;
            game.set_timecontrol(maintime * 100, 0, 0, 0);
        } else if (tc_type.find("canadian") != std::string::npos) {
            cmdstream >> maintime >> byotime >> byostones;
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);
        } else if (tc_type.find("byoyomi") != std::string::npos) {
            // KGS style Fischer clock
            cmdstream >> maintime >> byotime >> byoperiods;
            game.set_timecontrol(maintime * 100, byotime * 100, 0, byoperiods);
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return true;
        }

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return true;
    } else if (command.find("netbench") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int iterations;

        cmdstream >> tmp;  // eat netbench
        cmdstream >> iterations;

        if (!cmdstream.fail()) {
            Network::benchmark(&game, iterations);
        } else {
            Network::benchmark(&game);
        }
        gtp_printf(id, "");
        return true;

    } else if (command.find("printsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        cmdstream >> tmp;   // eat printsgf
        cmdstream >> filename;

        auto sgf_text = SGFTree::state_to_string(game, 0);

        if (cmdstream.fail()) {
            gtp_printf(id, "%s\n", sgf_text.c_str());
        } else {
            std::ofstream out(filename);
            out << sgf_text;
            out.close();
            gtp_printf(id, "");
        }

        return true;
    } else if (command.find("dump_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, winner_color, filename;
        int who_won;

        // tmp will eat "dump_training"
        cmdstream >> tmp >> winner_color >> filename;

        if (winner_color == "w" || winner_color == "white") {
            who_won = FullBoard::WHITE;
        } else if (winner_color == "b" || winner_color == "black") {
            who_won = FullBoard::BLACK;
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return true;
        }

        Training::dump_training(who_won, filename);
        filename += ".debug";
        Training::dump_stats(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    } else if (command.find("dump_supervised") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, winner_color, sgfname, outname;

        // tmp will eat dump_supervised
        cmdstream >> tmp >> sgfname >> outname;

        Training::dump_supervised(sgfname, outname);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return true;
    }

    gtp_fail_printf(id, "unknown command");
    return true;
}
