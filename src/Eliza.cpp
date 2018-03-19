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

#include <cstdlib>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include "eliza.h"

using namespace MB;

exchange_builder eliza::responds_to(const std::string& prompt) {
    return exchange_builder(*this, prompt);
}

void eliza::add_translations() {
    translations_.push_back(std::make_pair("am", "are"));
    translations_.push_back(std::make_pair("was", "were"));
    translations_.push_back(std::make_pair("i", "you"));
    translations_.push_back(std::make_pair("i'd", "you would"));
    translations_.push_back(std::make_pair("you'd", "I would"));
    translations_.push_back(std::make_pair("you're", "I am"));
    translations_.push_back(std::make_pair("i've", "you have"));
    translations_.push_back(std::make_pair("i'll", "you will"));
    translations_.push_back(std::make_pair("i'm", "you are"));
    translations_.push_back(std::make_pair("my", "your"));
    translations_.push_back(std::make_pair("are", "am"));
    translations_.push_back(std::make_pair("you've", "I have"));
    translations_.push_back(std::make_pair("you'll", "I will"));
    translations_.push_back(std::make_pair("your", "my"));
    translations_.push_back(std::make_pair("yours", "mine"));
    translations_.push_back(std::make_pair("you", "me"));
    translations_.push_back(std::make_pair("me", "you"));
    translations_.push_back(std::make_pair("myself", "yourself"));
    translations_.push_back(std::make_pair("yourself", "myself"));
}

std::string eliza::translate(const std::string& input) const {
    std::vector<std::string> words;
    boost::split(words, input, boost::is_any_of(" \t"));
    for (auto& word : words) {
        boost::algorithm::to_lower(word);
        for (auto& trans : translations_) {
            if (trans.first == word) {
                word = trans.second;
                break;
            }
        }
    }
    return boost::algorithm::join(words, " ");
}

std::string eliza::respond(const std::string& input) const {
    std::string response;
    for (auto& ex : exchanges_) {
        boost::smatch m;
        if (boost::regex_search(input, m, ex.prompt_)) {
            response = ex.responses_[rand() % ex.responses_.size()];
            if (m.size() > 1 && response.find("%1%") != response.npos) {
                std::string translation = translate(std::string(m[1].first, m[1].second));
                boost::trim_right_if(translation, boost::is_punct());
                response = boost::str(boost::format(response) % translation);
            }
            break;
        }
    }
    return response;
}

void eliza::add_responses() {
    responds_to("Can you (.*)")
        .with("Don't you believe that I can %1%?")
        .with("You want me to be able to %1%?");

    responds_to("Can I ([^\\?]*)\\??")
        .with("Perhaps you don't want to %1%.")
        .with("Do you want to be able to %1%?")
        .with("If you could %1%, would you?");

    responds_to("You are (.*)")
        .with("Why do you think I am %1%?")
        .with("Does it please you to think that I'm %1%?")
        .with("Perhaps you would like be %1%.")
        .with("Do you sometimes wish you were %1%?");

    responds_to("I don'?t (.*)")
        .with("Don't you really %1%?")
        .with("Why don't you %1%?")
        .with("Do you wish to be able to %1%?");

    responds_to("I feel (.*)")
        .with("Does it trouble you to feel %1%?")
        .with("Do you often feel %1%?")
        .with("Do you enjoy feeling %?%")
        .with("When do you usually feel %1%?")
        .with("When you feel %1%, what do you do?");

    responds_to("Why don'?t you ([^\\?]*)\\??")
        .with("Do you really believe I don't %1%?")
        .with("Perhaps in good time I will %1%.")
        .with("Do you want me to %1%?");

    responds_to("Why can'?t I ([^\\?]*)\\??")
        .with("Do you think you should be able to %1%?")
        .with("Why can't you?");

    responds_to("Are you ([^\\?]*)\\??")
        .with("Why are you interested in whether I am %1%?")
        .with("Would you prefer it if I were not %1%?")
        .with("Perhaps in your fantasies I am %1%.");

    responds_to("I can'?t (.*)")
        .with("How do you know you can't %1%?")
        .with("Have you tried to %1%?")
        .with("Perhaps now you can %1%");

    responds_to("I am (.*)")
        .with("Did you come to me because you are %1%?")
        .with("How long have you been %1%?");

    responds_to("I'?m (.*)")
        .with("Do you believe it is normal to be %1%?")
        .with("Do you enjoy being %1%?");

    responds_to("You (.*)")
        .with("We were discussing you - not me.")
        .with("Oh, I %1%?");

    responds_to("I want (.*)")
        .with("What would it mean to you if you got %1%?")
        .with("Why do you want %1%?")
        .with("Suppose you soon got %1%?")
        .with("What if you never got %1%?")
        .with("I sometimes also want %1%");

    responds_to("What (.*)")
        .with("Why do you ask?")
        .with("Does that question interest you?")
        .with("What answer would please you the most?")
        .with("What do you think?")
        .with("Are such questions on your mind often?")
        .with("What is it that you really want to know?")
        .with("Have you asked anyone else?")
        .with("Have you asked such questions before?")
        .with("What else comes to mind when you ask that?");

    responds_to("Because")
        .with("Is that the real reason?")
        .with("Don't any other reasons come to mind?")
        .with("Does that reason explain anything else?")
        .with("What other reasons might there be?");

    responds_to("Sorry")
        .with("There are many times when no apology is needed.")
        .with("What feelings do you have when you apologize?");

    responds_to("Dream")
        .with("What does that dream suggest to you?")
        .with("Do you dream often?")
        .with("What persons appear in your dreams?")
        .with("Are you disturbed by your dreams?");

    responds_to("^Hello")
        .with("How do you do... Please state your problem.");

    responds_to("^Maybe")
        .with("You don't seem quite certain.")
        .with("Why the uncertain tone?")
        .with("Can't you be more positive?")
        .with("You aren't sure?")
        .with("Don't you know?");

    responds_to("^No[.!]?$")
        .with("Are you saying that just to be negative?")
        .with("You are being a bit negative.")
        .with("Why not?")
        .with("Are you sure?")
        .with("Why no?");

    responds_to("Your (.*)")
        .with("Why are you concerned about my %1%?")
        .with("What about your own %1%?");

    responds_to("Always")
        .with("Can you think of a specific example?")
        .with("When?")
        .with("Really, always?");

    responds_to("Think (.*)")
        .with("What are you thinking of?")
        .with("Do you really think so?")
        .with("But you are not sure %1%?")
        .with("Do you doubt %1%?");

    responds_to("Alike")
        .with("In what way?")
        .with("What resemblance do you see?")
        .with("What does the similarity suggest to you?")
        .with("what other connections do you see?")
        .with("Could there really be some connection?")
        .with("How?");

    responds_to("^Yes[.!]?$")
        .with("You seem quite positive.")
        .with("Are you sure?")
        .with("I see.")
        .with("I understand.");

    responds_to("Friend")
        .with("Why do you bring up the topic of friends?")
        .with("Do your friends worry you?")
        .with("Do your friends pick on you?")
        .with("Are you sure you have any friends?")
        .with("Do you impose on your friends?")
        .with("Perhaps your love for friends worries you.");

    responds_to("Computer")
        .with("Do computers worry you?")
        .with("Are you talking about me in particular?")
        .with("Are you frightened by machines?")
        .with("Why do you mention computers?")
        .with("What do you think machines have to do with your problem?")
        .with("Don't you think computers can help people?")
        .with("What is it about machines that worries you?");

    responds_to("(.*)")
        .with("Say, do you have any psychological problems?")
        .with("What does that suggest to you?")
        .with("I see.")
        .with("I'm not sure I understand you fully.")
        .with("Come come elucidate your thoughts.")
        .with("Can you elaborate on that?")
        .with("That is quite interesting.");
}