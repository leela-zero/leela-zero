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
