/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Gian-Carlo Pascutto and contributors

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
#include <gtest/gtest.h>

#include "config.h"

#include <cstdint>
#include <algorithm>
#include <iostream>
#include <memory>
#include <regex>
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

void expect_regex(std::string s, std::string re, bool positive = true) {
    auto m = std::regex_search(s, std::regex(re));
    if (positive && !m) {
        FAIL() << "Output:" << std::endl << s
            << "Does not contain:" << std::endl
            << re << std::endl;
    } else if (!positive && m) {
        FAIL() << "output:" << std::endl << s
            << "Should not contain:" << std::endl
            << re << std::endl;
    }
}

class LeelaEnv: public ::testing::Environment {
public:
    ~LeelaEnv() {}
    void SetUp() {
        GTP::setup_default_parameters();
        cfg_gtp_mode = true;

        // Setup global objects after command line has been parsed
        thread_pool.initialize(cfg_num_threads);

        // Use deterministic random numbers for hashing
        auto rng = std::make_unique<Random>(5489);
        Zobrist::init_zobrist(*rng);

        // Initialize the main thread RNG.
        // Doing this here avoids mixing in the thread_id, which
        // improves reproducibility across platforms.
        Random::get_Rng().seedrandom(cfg_rng_seed);

        cfg_weightsfile = "../src/tests/0k.txt";

        auto playouts = std::min(cfg_max_playouts, cfg_max_visits);
        auto network = std::make_unique<Network>();
        network->initialize(playouts, cfg_weightsfile);
        GTP::initialize(std::move(network));
    }
    void TearDown() {}
};

::testing::Environment* const leela_env = ::testing::AddGlobalTestEnvironment(new LeelaEnv);

class LeelaTest: public ::testing::Test {
public:
    LeelaTest() {
        // Reset engine parameters
        GTP::setup_default_parameters();
        cfg_max_playouts = 1;
        cfg_gtp_mode = true;

        m_gamestate = std::make_unique<GameState>();
        m_gamestate->init_game(19, 7.5f);
    }

    GameState& get_gamestate() {
        return *m_gamestate;
    }
    std::pair<std::string, std::string> gtp_execute(std::string cmd) {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        GTP::execute(get_gamestate(), cmd);
        return std::make_pair(testing::internal::GetCapturedStdout(),
                              testing::internal::GetCapturedStderr());
    }
    void test_analyze_cmd(std::string cmd, bool valid, int who, int interval,
            int avoidlen, int avoidcolor, int avoiduntil);

private:
    std::unique_ptr<GameState> m_gamestate;
};

TEST_F(LeelaTest, Startup) {
    auto maingame = get_gamestate();
}

TEST_F(LeelaTest, DefaultHash) {
    auto maingame = get_gamestate();
    auto hash = maingame.board.get_hash();
    auto ko_hash = maingame.board.get_ko_hash();

    EXPECT_EQ(hash, 0x9A930BE1616C538E);
    EXPECT_EQ(ko_hash, 0xA14C933E7669946D);
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

TEST_F(LeelaTest, KoPntNotSame) {
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
    // But ko (intersection) is not
    EXPECT_NE(hash, maingame.board.get_hash());
}

TEST_F(LeelaTest, MoveOnOccupiedPnt) {
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

// Basic TimeControl test
TEST_F(LeelaTest, TimeControl) {
    std::pair<std::string, std::string> result;

    // clear_board to force GTP to make a new UCTSearch.
    // This will pickup our new cfg_* settings.
    result = gtp_execute("clear_board");

    result = gtp_execute("kgs-time_settings canadian 0 120 25");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: 00:02:00, 25 stones left");
    expect_regex(result.second, "White time: 00:02:00, 25 stones left");

    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: \\S*, 24 stones left");
    expect_regex(result.second, "White time: \\S*, 25 stones left");

    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: \\S*, 24 stones left");
    expect_regex(result.second, "White time: \\S*, 24 stones left");
}

// Test changing TimeControl during game
TEST_F(LeelaTest, TimeControl2) {
    std::pair<std::string, std::string> result;

    // clear_board to force GTP to make a new UCTSearch.
    // This will pickup our new cfg_* settings.
    result = gtp_execute("clear_board");

    result = gtp_execute("kgs-time_settings byoyomi 0 100 1");
    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: 00:01:40, 1 period\\(s\\) of 100 seconds left");
    expect_regex(result.second, "White time: 00:01:40, 1 period\\(s\\) of 100 seconds left");

    result = gtp_execute("kgs-time_settings byoyomi 0 120 1");
    result = gtp_execute("go");
    result = gtp_execute("showboard");
    expect_regex(result.second, "Black time: 00:02:00, 1 period\\(s\\) of 120 seconds left");
    expect_regex(result.second, "White time: 00:02:00, 1 period\\(s\\) of 120 seconds left");
}

void LeelaTest::test_analyze_cmd(std::string cmd, bool valid, int who, int interval,
        int avoidlen, int avoidcolor, int avoiduntil) {
    // std::cout << "testing " << cmd << std::endl;
    // avoid_until checks against the absolute game move number, indexed from 0
    std::istringstream cmdstream(cmd);
    auto maingame = get_gamestate();
    AnalyzeTags result{cmdstream, maingame};
    EXPECT_EQ(result.m_invalid, !valid);
    if (!valid) return;
    EXPECT_EQ(result.m_who, who);
    EXPECT_EQ(result.m_interval_centis, interval);
    EXPECT_EQ(result.m_moves_to_avoid.size(), avoidlen);
    if (avoidlen) {
        EXPECT_EQ(result.m_moves_to_avoid[0].color, avoidcolor);
        EXPECT_EQ(result.m_moves_to_avoid[0].until_move, avoiduntil);
    }
}

// Test parsing the lz-analyze command line
TEST_F(LeelaTest, AnalyzeParse) {
    gtp_execute("clear_board");

    test_analyze_cmd("b 50",
            true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("50 b",
            true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("b interval 50",
            true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("interval 50 b",
            true, FastBoard::BLACK, 50, 0, -1, -1);
    test_analyze_cmd("b interval",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("42 w",
            true, FastBoard::WHITE, 42, 0, -1, -1);
    test_analyze_cmd("1234",
            true, FastBoard::BLACK, 1234, 0, -1, -1);
    gtp_execute("play b q16");
    test_analyze_cmd("1234",
            true, FastBoard::WHITE, 1234, 0, -1, -1);
    test_analyze_cmd("b 100 avoid b k10 1",
            true, FastBoard::BLACK, 100, 1, FastBoard::BLACK, 1);
    test_analyze_cmd("b 100 avoid b k10 1 avoid b a1 1",
            true, FastBoard::BLACK, 100, 2, FastBoard::BLACK, 1);
    test_analyze_cmd("b 100 avoid w k10 8",
            true, FastBoard::BLACK, 100, 1, FastBoard::WHITE, 8);
    gtp_execute("play w q4");
    test_analyze_cmd("b 100 avoid b k10 8",
            true, FastBoard::BLACK, 100, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("100 b avoid b k10 8",
            true, FastBoard::BLACK, 100, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("b avoid b k10 8 100",
            true, FastBoard::BLACK, 100, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b k10 8 100 b",
            true, FastBoard::BLACK, 100, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b k10 8 100 w",
            true, FastBoard::WHITE, 100, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b z10 8 100 w",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("avoid b k10 8 100 w bogus",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("avoid b k10 8 100 w avoid b pass 17",
            true, FastBoard::WHITE, 100, 2, FastBoard::BLACK, 9);
    test_analyze_cmd("avoid b k10 8 w avoid b pass 17",
            true, FastBoard::WHITE, 0, 2, FastBoard::BLACK, 9);

    gtp_execute("clear_board");
    test_analyze_cmd("b avoid b a1 10 allow b t1 1",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid w a1 10 allow b t1 1",
            true, FastBoard::BLACK, 0, 1, FastBoard::WHITE, 9);
    test_analyze_cmd("b avoid b pass 10 allow b t1 1",
            true, FastBoard::BLACK, 0, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("b avoid b resign 10 allow b t1 1",
            true, FastBoard::BLACK, 0, 1, FastBoard::BLACK, 9);
    test_analyze_cmd("b avoid w c3,c4,d3,d4 2 avoid b pass 50",
            true, FastBoard::BLACK, 0, 5, FastBoard::WHITE, 1);
    test_analyze_cmd("b avoid w c3,c4,d3,d4, 2 avoid b pass 50",
            false, -1, -1, -1, -1, -1);

    gtp_execute("clear_board");
    test_analyze_cmd("b avoid b q16 1",
            true, FastBoard::BLACK, 0, 1, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b : 1",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b d4: 1",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b d14: 1",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b :e3 1",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b d:e3 1",
            false, -1, -1, -1, -1, -1);
    test_analyze_cmd("b avoid b q16:q16 20",
            true, FastBoard::BLACK, 0, 1, FastBoard::BLACK, 19);
    test_analyze_cmd("b avoid b q16:t19 1",
            true, FastBoard::BLACK, 0, 16, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b t19:q16 1",
            true, FastBoard::BLACK, 0, 16, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b t16:q19 1",
            true, FastBoard::BLACK, 0, 16, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b q19:t16 1",
            true, FastBoard::BLACK, 0, 16, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b a1:t19 1",
            true, FastBoard::BLACK, 0, 361, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b a1:t19 1 avoid w pass 1 avoid w resign 1",
            true, FastBoard::BLACK, 0, 363, FastBoard::BLACK, 0);
    test_analyze_cmd("b avoid b a1:t19,pass,resign 1",
            true, FastBoard::BLACK, 0, 363, FastBoard::BLACK, 0);
}
