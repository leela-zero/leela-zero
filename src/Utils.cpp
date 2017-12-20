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
#include "Utils.h"

#include <mutex>
#include <stdarg.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/select.h>
#endif

#include "GTP.h"

Utils::ThreadPool thread_pool;

bool Utils::input_pending(void) {
#ifdef HAVE_SELECT
    fd_set read_fds;
    struct timeval timeout;
    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    timeout.tv_sec = timeout.tv_usec = 0;
    select(1,&read_fds,NULL,NULL,&timeout);
    if (FD_ISSET(0,&read_fds)) {
        return true;
    } else {
        return false;
    }
#else
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;

    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }

    if (pipe) {
        if (!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        if (dw) {
            return true;
        } else {
            return false;
        }
    } else {
        if (!GetNumberOfConsoleInputEvents(inh, &dw)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        if (dw <= 1) {
            return false;
        } else {
            return true;
        }
    }
    return false;
#endif
}

static std::mutex IOmutex;

void Utils::myprintf(const char *fmt, ...) {
    if (cfg_quiet) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
    }
}

void Utils::gtp_printf(int id, const char *fmt, ...) {
    va_list ap;

    if (id != -1) {
        fprintf(stdout, "=%d ", id);
    } else {
        fprintf(stdout, "= ");
    }

    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    printf("\n\n");

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        if (id != -1) {
            fprintf(cfg_logfile_handle, "=%d ", id);
        } else {
            fprintf(cfg_logfile_handle, "= ");
        }
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
        fprintf(cfg_logfile_handle, "\n\n");
    }
}

void Utils::gtp_fail_printf(int id, const char *fmt, ...) {
    va_list ap;

    if (id != -1) {
        fprintf(stdout, "?%d ", id);
    } else {
        fprintf(stdout, "? ");
    }

    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    printf("\n\n");

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        if (id != -1) {
            fprintf(cfg_logfile_handle, "?%d ", id);
        } else {
            fprintf(cfg_logfile_handle, "? ");
        }
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
        fprintf(cfg_logfile_handle, "\n\n");
    }
}

void Utils::log_input(std::string input) {
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        fprintf(cfg_logfile_handle, ">>%s\n", input.c_str());
    }
}
