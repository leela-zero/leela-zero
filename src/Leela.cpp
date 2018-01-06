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

#include "config.h"

#include <cstdint>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

static void license_blurb() {
    printf(
        "Leela Zero  Copyright (C) 2017  Gian-Carlo Pascutto\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the COPYING file for details.\n\n"
    );
}

static void parse_commandline(int argc, char *argv[], bool & gtp_mode) {
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description v_desc("Allowed options");
    v_desc.add_options()
        ("help,h", "Show commandline options.")
        ("gtp,g", "Enable GTP mode.")
        ("threads,t", po::value<int>()->default_value
                      (std::min(2, cfg_num_threads)),
                      "Number of threads to use.")
        ("playouts,p", po::value<int>(),
                       "Weaken engine by limiting the number of playouts. "
                       "Requires --noponder.")
        ("lagbuffer,b", po::value<int>()->default_value(cfg_lagbuffer_cs),
                        "Safety margin for time usage in centiseconds.")
        ("resignpct,r", po::value<int>()->default_value(cfg_resignpct),
                        "Resign when winrate is less than x%.")
        ("randomcnt,m", po::value<int>()->default_value(cfg_random_cnt),
                        "Play more randomly the first x moves.")
        ("noise,n", "Enable policy network randomization.")
        ("seed,s", po::value<std::uint64_t>(),
                   "Random number generation seed.")
        ("dumbpass,d", "Don't use heuristics for smarter passing.")
        ("weights,w", po::value<std::string>(), "File with network weights.")
        ("logfile,l", po::value<std::string>(), "File to log input/output to.")
        ("quiet,q", "Disable all diagnostic output.")
        ("noponder", "Disable thinking on opponent's time.")
#ifdef USE_OPENCL
        ("gpu",  po::value<std::vector<int> >(),
                "ID of the OpenCL device(s) to use (disables autodetection).")
        ("rowtiles", po::value<int>()->default_value(cfg_rowtiles),
                     "Split up the board in # tiles.")
#endif
#ifdef USE_TUNER
        ("puct", po::value<float>())
        ("softmax_temp", po::value<float>())
#endif
        ;
    // These won't be shown, we use them to catch incorrect usage of the
    // command line.
    po::options_description h_desc("Hidden options");
    h_desc.add_options()
        ("arguments", po::value<std::vector<std::string>>());
    // Parse both the above, we will check if any of the latter are present.
    po::options_description all("All options");
    all.add(v_desc).add(h_desc);
    po::positional_options_description p_desc;
    p_desc.add("arguments", -1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                  .options(all).positional(p_desc).run(), vm);
        po::notify(vm);
    }  catch(const boost::program_options::error& e) {
        myprintf("ERROR: %s\n", e.what());
        license_blurb();
        std::cout << v_desc << std::endl;
        exit(EXIT_FAILURE);
    }

    // Handle commandline options
    if (vm.count("help") || vm.count("arguments")) {
        auto ev = EXIT_SUCCESS;
        // The user specified an argument. We don't accept any, so explain
        // our usage.
        if (vm.count("arguments")) {
            for (auto& arg : vm["arguments"].as<std::vector<std::string>>()) {
                std::cout << "Unrecognized argument: " << arg << std::endl;
            }
            ev = EXIT_FAILURE;
        }
        license_blurb();
        std::cout << v_desc << std::endl;
        exit(ev);
    }

    if (vm.count("quiet")) {
        cfg_quiet = true;
    }

#ifdef USE_TUNER
    if (vm.count("puct")) {
        cfg_puct = vm["puct"].as<float>();
    }
    if (vm.count("softmax_temp")) {
        cfg_softmax_temp = vm["softmax_temp"].as<float>();
    }
#endif

    if (vm.count("logfile")) {
        cfg_logfile = vm["logfile"].as<std::string>();
        myprintf("Logging to %s.\n", cfg_logfile.c_str());
        cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
    }

    if (vm.count("weights")) {
        cfg_weightsfile = vm["weights"].as<std::string>();
    } else {
        myprintf("A network weights file is required to use the program.\n");
        exit(EXIT_FAILURE);
    }

    if (vm.count("gtp")) {
        gtp_mode = true;
    }

    if (vm.count("threads")) {
        int num_threads = vm["threads"].as<int>();
        if (num_threads > cfg_num_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_num_threads);
        } else if (num_threads != cfg_num_threads) {
            myprintf("Using %d thread(s).\n", num_threads);
            cfg_num_threads = num_threads;
        }
    }

    if (vm.count("seed")) {
        cfg_rng_seed = vm["seed"].as<std::uint64_t>();
        if (cfg_num_threads > 1) {
            myprintf("Seed specified but multiple threads enabled.\n");
            myprintf("Games will likely not be reproducible.\n");
        }
    }
    myprintf("RNG seed: %llu\n", cfg_rng_seed);

    if (vm.count("noponder")) {
        cfg_allow_pondering = false;
    }

    if (vm.count("noise")) {
        cfg_noise = true;
    }

    if (vm.count("dumbpass")) {
        cfg_dumbpass = true;
    }

    if (vm.count("playouts")) {
        cfg_max_playouts = vm["playouts"].as<int>();
        if (!vm.count("noponder")) {
            myprintf("Nonsensical options: Playouts are restricted but "
                     "thinking on the opponent's time is still allowed. "
                     "Add --noponder if you want a weakened engine.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (vm.count("resignpct")) {
        cfg_resignpct = vm["resignpct"].as<int>();
    }

    if (vm.count("randomcnt")) {
        cfg_random_cnt = vm["randomcnt"].as<int>();
    }

    if (vm.count("lagbuffer")) {
        int lagbuffer = vm["lagbuffer"].as<int>();
        if (lagbuffer != cfg_lagbuffer_cs) {
            myprintf("Using per-move time margin of %.2fs.\n", lagbuffer/100.0f);
            cfg_lagbuffer_cs = lagbuffer;
        }
    }

#ifdef USE_OPENCL
    if (vm.count("gpu")) {
        cfg_gpus = vm["gpu"].as<std::vector<int> >();
    }

    if (vm.count("rowtiles")) {
        int rowtiles = vm["rowtiles"].as<int>();
        rowtiles = std::min(19, rowtiles);
        rowtiles = std::max(1, rowtiles);
        if (rowtiles != cfg_rowtiles) {
            myprintf("Splitting the board in %d tiles.\n", rowtiles);
            cfg_rowtiles = rowtiles;
        }
    }
#endif
}

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
    Network::initialize();
}

int main (int argc, char *argv[]) {
    bool gtp_mode = false;
    std::string input;

    // Set up engine parameters
    GTP::setup_default_parameters();
    parse_commandline(argc, argv, gtp_mode);

    // Disable IO buffering as much as possible
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    std::cin.setf(std::ios::unitbuf);

    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
#ifndef WIN32
    setbuf(stdin, nullptr);
#endif

    if (!gtp_mode) {
        license_blurb();
    }

    init_global_objects();

    auto maingame = std::make_unique<GameState>();

    /* set board limits */
    float komi = 7.5;
    maingame->init_game(19, komi);

    for(;;) {
        if (!gtp_mode) {
            maingame->display_state();
            std::cout << "Leela: ";
        }

        if (std::getline(std::cin, input)) {
            Utils::log_input(input);
            GTP::execute(*maingame, input);
        } else {
            // eof or other error
            break;
        }

        // Force a flush of the logfile
        if (cfg_logfile_handle) {
            fclose(cfg_logfile_handle);
            cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
        }
    }

    return 0;
}
