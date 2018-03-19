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

#ifndef ELIZA_HPP
#define ELIZA_HPP

#include <string>
#include <vector>
#include <boost/regex.hpp>

#include "config.h"

namespace MB {

    struct exchange {
        boost::regex prompt_;
        std::vector<std::string> responses_;
        explicit exchange(const std::string& prompt)
            : prompt_(prompt, boost::regex::icase) {
        }
    };

    class exchange_builder;

    class eliza {
        std::string name_;
        std::vector<exchange> exchanges_;
        std::vector<std::pair<std::string, std::string> > translations_;
    public:
        eliza(const std::string& name = PROGRAM_NAME) : name_(name) {
            add_translations();
            add_responses();
        }
        const std::string& name() const {
            return name_;
        }
        exchange_builder responds_to(const std::string& prompt);
        void add_exchange(const exchange& ex) {
            exchanges_.push_back(ex);
        }
        std::string respond(const std::string& input) const;
    private:
        void add_translations();
        void add_responses();
        std::string translate(const std::string& input) const;
    };

    class exchange_builder {
        friend eliza;
        eliza& eliza_;
        exchange exchange_;

        exchange_builder(eliza& el, const std::string& prompt) : eliza_(el), exchange_(prompt) {
        }

    public:
        ~exchange_builder() {
            eliza_.add_exchange(exchange_);
        }

        exchange_builder& with(const std::string& response) {
            exchange_.responses_.push_back(response);
            return *this;
        }
    };

}; // namespace MB

#endif // ELIZA_HPP
