/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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
#include <cstdarg>
#include <cstdio>

#include <boost/filesystem.hpp>

#ifdef _WIN32
#include <windows.h>
#include <shlwapi.h>
#pragma comment(lib,"shlwapi.lib")
#include "shlobj.h"
#else
#include <sys/select.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

#include "GTP.h"

Utils::ThreadPool thread_pool;

bool Utils::input_pending(void) {
#ifdef HAVE_SELECT
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    struct timeval timeout{0,0};
    select(1,&read_fds,nullptr,nullptr,&timeout);
    return FD_ISSET(0, &read_fds);
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
        if (!PeekNamedPipe(inh, nullptr, 0, nullptr, &dw, nullptr)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw;
    } else {
        if (!GetNumberOfConsoleInputEvents(inh, &dw)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw > 1;
    }
    return false;
#endif
}

static std::mutex IOmutex;

void Utils::myprintf(const char *fmt, ...) {
    if (cfg_quiet) {
        return;
    }
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

static void gtp_fprintf(FILE* file, const std::string& prefix,
                        const char *fmt, va_list ap) {
    fprintf(file, "%s ", prefix.c_str());
    vfprintf(file, fmt, ap);
    fprintf(file, "\n\n");
}

static void gtp_base_printf(int id, std::string prefix,
                            const char *fmt, va_list ap) {
    if (id != -1) {
        prefix += std::to_string(id);
    }
    gtp_fprintf(stdout, prefix, fmt, ap);
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        gtp_fprintf(cfg_logfile_handle, prefix, fmt, ap);
    }
}

void Utils::gtp_printf(int id, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "=", fmt, ap);
    va_end(ap);
}

void Utils::gtp_printf_raw(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
    }
}

void Utils::gtp_fail_printf(int id, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "?", fmt, ap);
    va_end(ap);
}

void Utils::log_input(const std::string& input) {
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        fprintf(cfg_logfile_handle, ">>%s\n", input.c_str());
    }
}

size_t Utils::ceilMultiple(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

#ifdef _WIN32
// https://stackoverflow.com/a/3999597
// Convert a wide Unicode string to an UTF8 string
// Thanks Windows, saving developers time since 1995
std::string utf8_encode(const std::wstring &wstr)
{
    if ( wstr.empty() ) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}
#endif

std::string Utils::leelaz_file(std::string file) {
#ifdef _WIN32
    // https://stackoverflow.com/a/11032792 but avoiding the UNICODE mess and just using W variants
    wchar_t szPath[MAX_PATH];
    if (!SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_COMMON_APPDATA, NULL, 0, szPath)))
        return NULL;
    PathAppendW(szPath, L"\\Leela Zero\\");
    boost::filesystem::path dir(utf8_encode(szPath));
#else
    // https://stackoverflow.com/a/26696759
    const char *homedir;
    if ((homedir = getenv("HOME")) == NULL) {
        struct passwd *pwd;
        if ((pwd = getpwuid(getuid())) == NULL) { // NOLINT(runtime/threadsafe_fn)
            return NULL;
        }
        homedir = pwd->pw_dir;
    }
    boost::filesystem::path dir(homedir);
    dir /= ".local/share/leela-zero";
#endif
    boost::filesystem::create_directories(dir);
    dir /= file;
    return dir.string();
}
