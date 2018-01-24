/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Gian-Carlo Pascutto

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
#include <gtest/gtest.h>

#include "config.h"

#include <cstdint>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "GTP.h"
#include "GameState.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

// Setup global objects after command line has been parsed
void init_global_objects() {
    thread_pool.initialize(cfg_num_threads);

    // Use deterministic random numbers for hashing
    auto rng = std::make_unique<Random>(5489);
    Zobrist::init_zobrist(*rng);

    // Initialize the main thread RNG.
    // Doing this here avoids mixing in the thread_id, which
    // improves reproducibility across platforms.
    Random::get_Rng().seedrandom(cfg_rng_seed);

    NNCache::get_NNCache().set_size_from_playouts(cfg_max_playouts);

    // Initialize network
    // Needs a weights file
    // Network::initialize();
}

class LeelaTest: public ::testing::Test {
public:
    LeelaTest( ) {
        // Setup engine parameters
        GTP::setup_default_parameters();
        cfg_gtp_mode = true;
        init_global_objects();

        m_gamestate = std::make_unique<GameState>();
        m_gamestate->init_game(19, 7.5f);
    }

    GameState& get_gamestate() {
        return *m_gamestate;
    }

    std::unique_ptr<GameState> m_gamestate;
};

TEST_F(LeelaTest, Startup) {
    auto maingame = get_gamestate();
}

TEST_F(LeelaTest, DefaultHash) {
    auto maingame = get_gamestate();
    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    EXPECT_EQ(hash, 0x30C547108A9AF65FULL);
    EXPECT_EQ(ko_hash, 0x9EC2A5B7968B5F23ULL);
}

TEST_F(LeelaTest, Transposition) {
    auto maingame = get_gamestate();

    testing::internal::CaptureStdout();
    GTP::execute(maingame, "play b Q16");
    GTP::execute(maingame, "play w D16");
    GTP::execute(maingame, "play b D4");

    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    GTP::execute(maingame, "clear_board");

    GTP::execute(maingame, "play b D4");
    GTP::execute(maingame, "play w D16");
    GTP::execute(maingame, "play b Q16");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(hash, maingame.board.get_hash());
    EXPECT_EQ(ko_hash, maingame.board.get_ko_hash());
}

TEST_F(LeelaTest, KoSqNotSame) {
    auto maingame = get_gamestate();

    testing::internal::CaptureStdout();
    GTP::execute(maingame, "play b E6");
    GTP::execute(maingame, "play w F6");
    GTP::execute(maingame, "play b E5");
    GTP::execute(maingame, "play w F5");
    GTP::execute(maingame, "play b D4");
    GTP::execute(maingame, "play w E4");
    GTP::execute(maingame, "play b E3");
    GTP::execute(maingame, "play w G4");
    GTP::execute(maingame, "play b F4"); // capture
    GTP::execute(maingame, "play w F3");
    GTP::execute(maingame, "play b D3");

    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    GTP::execute(maingame, "clear_board");

    GTP::execute(maingame, "play b E6");
    GTP::execute(maingame, "play w F6");
    GTP::execute(maingame, "play b E5");
    GTP::execute(maingame, "play w F5");
    GTP::execute(maingame, "play b D4");
    GTP::execute(maingame, "play w E4");
    GTP::execute(maingame, "play b E3");
    GTP::execute(maingame, "play w G4");
    GTP::execute(maingame, "play b D3");
    GTP::execute(maingame, "play w F3");
    GTP::execute(maingame, "play b F4"); // capture
    std::string output = testing::internal::GetCapturedStdout();

    // Board position is the same
    EXPECT_EQ(ko_hash, maingame.board.get_ko_hash());
    // But ko (square) is not
    EXPECT_NE(hash, maingame.board.get_hash());
}

TEST_F(LeelaTest, MoveOnOccupiedSq) {
    auto maingame = get_gamestate();
    std::string output;

    {
        testing::internal::CaptureStdout();
        GTP::execute(maingame, "play b D4");
        GTP::execute(maingame, "play b D4");
        output = testing::internal::GetCapturedStdout();
    }

    // Find this error in the output
    EXPECT_NE(output.find("illegal move"), std::string::npos);

    {
        testing::internal::CaptureStdout();
        GTP::execute(maingame, "play w Q16");
        GTP::execute(maingame, "play b Q16");
        output = testing::internal::GetCapturedStdout();
    }

    // Find this error in the output
    EXPECT_NE(output.find("illegal move"), std::string::npos);
}


TEST_F(LeelaTest, TimeControl) {
    // Initialize network
    cfg_weightsfile = "../src/tests/0k.txt";
    Network::initialize();
    cfg_max_playouts = 1;
    cfg_num_threads = 1;
    std::string output;

    // TODO I tried to make these separate tests,
    // but it died locally when leelaz ran the second time.
    {
        auto maingame = get_gamestate();
        // clear_board to force GTP to make a new UCTSearch.
        // This will pickup our new cfg_* settings.
        GTP::execute(maingame, "clear_board");

        GTP::execute(maingame, "kgs-time_settings byoyomi 0 100 1");
        GTP::execute(maingame, "go");
        testing::internal::CaptureStderr();
        GTP::execute(maingame, "showboard");
        output = testing::internal::GetCapturedStderr();
        EXPECT_NE(output.find("Black time: 00:01:40, 1 period(s) of 100 seconds left"), std::string::npos);
        EXPECT_NE(output.find("White time: 00:01:40, 1 period(s) of 100 seconds left"), std::string::npos);

        GTP::execute(maingame, "kgs-time_settings byoyomi 0 120 1");
        GTP::execute(maingame, "go");
        testing::internal::CaptureStderr();
        GTP::execute(maingame, "showboard");
        output = testing::internal::GetCapturedStderr();
        EXPECT_NE(output.find("Black time: 00:02:00, 1 period(s) of 120 seconds left"), std::string::npos);
        EXPECT_NE(output.find("White time: 00:02:00, 1 period(s) of 120 seconds left"), std::string::npos);
    }

    {
        cfg_max_playouts = 0;
        auto maingame = get_gamestate();
        // clear_board to force GTP to make a new UCTSearch.
        // This will pickup our new cfg_* settings.
        GTP::execute(maingame, "clear_board");

        // Absolute time 100s = 1.37s per move in opening.
        // Enough to be visible on display_times
        GTP::execute(maingame, "time_settings 100 0 0");
        // Use assert here because if the time breaks we might
        // cause the test to think forever.
        testing::internal::CaptureStderr();
        GTP::execute(maingame, "showboard");
        output = testing::internal::GetCapturedStderr();
        printf("output\n%s\noutput", output.c_str());
        EXPECT_NE(output.find("Black time: 00:01:40"), std::string::npos);
        EXPECT_NE(output.find("White time: 00:01:40"), std::string::npos);

        GTP::execute(maingame, "go");
        testing::internal::CaptureStderr();
        GTP::execute(maingame, "showboard");
        output = testing::internal::GetCapturedStderr();
        printf("output\n%s\noutput", output.c_str());
        EXPECT_NE(output.find("Black time: 00:01:38"), std::string::npos);
        EXPECT_NE(output.find("White time: 00:01:40"), std::string::npos);

        GTP::execute(maingame, "go");
        testing::internal::CaptureStderr();
        GTP::execute(maingame, "showboard");
        output = testing::internal::GetCapturedStderr();
        printf("output\n%s\noutput", output.c_str());
        EXPECT_NE(output.find("Black time: 00:01:38"), std::string::npos);
        EXPECT_NE(output.find("White time: 00:01:38"), std::string::npos);
    }
}
