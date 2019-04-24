/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Marco Calignano

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

#include <cmath>
#include <random>
#include <QDir>
#include <QThread>
#include <QList>
#include <QCryptographicHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLockFile>
#include <QUuid>
#include <QRegularExpression>
#include <QVariant>
#include "Management.h"
#include "Game.h"


constexpr int RETRY_DELAY_MIN_SEC = 30;
constexpr int RETRY_DELAY_MAX_SEC = 60 * 60;  // 1 hour
constexpr int MAX_RETRIES = 3;           // Stop retrying after 3 times

const QString server_url = "https://zero.sjeng.org/";
const QString Leelaz_min_version = "0.12";

Management::Management(const int gpus,
                       const int games,
                       const QStringList& gpuslist,
                       const int ver,
                       const int maxGames,
                       const bool delNetworks,
                       const QString& keep,
                       const QString& debug)

    : m_syncMutex(),
    m_gamesThreads(gpus * games),
    m_games(games),
    m_gpus(gpus),
    m_gpusList(gpuslist),
    m_selfGames(0),
    m_matchGames(0),
    m_gamesPlayed(0),
    m_keepPath(keep),
    m_debugPath(debug),
    m_version(ver),
    m_fallBack(Order::Error),
    m_lastMatch(Order::Error),
    m_gamesLeft(maxGames),
    m_threadsLeft(gpus * games),
    m_delNetworks(delNetworks),
    m_lockFile(nullptr) {
}

void Management::runTuningProcess(const QString &tuneCmdLine) {
    QTextStream(stdout) << tuneCmdLine << endl;
    QProcess tuneProcess;
    tuneProcess.start(tuneCmdLine);
    tuneProcess.waitForStarted(-1);
    while (tuneProcess.state() == QProcess::Running) {
        tuneProcess.waitForReadyRead(1000);
        QByteArray text = tuneProcess.readAllStandardOutput();
        int version_start = text.indexOf("Leela Zero ") + 11;
        if (version_start > 10) {
            int version_end = text.indexOf(" ", version_start);
            m_leelaversion = QString(text.mid(version_start, version_end - version_start));
        }
        QTextStream(stdout) << text;
        QTextStream(stdout) << tuneProcess.readAllStandardError();
    }
    QTextStream(stdout) << "Found Leela Version : " << m_leelaversion << endl;
    tuneProcess.waitForFinished(-1);
}

Order Management::getWork(const QFileInfo &file) {
    QTextStream(stdout) << "Got previously stored file" <<endl;
    Order o;
    o.load(file.fileName());
    QFile::remove(file.fileName());
    m_lockFile->unlock();
    delete m_lockFile;
    m_lockFile = nullptr;
    return o;
}

void Management::giveAssignments() {
    sendAllGames();

    //Make the OpenCl tuning before starting the threads
    QTextStream(stdout) << "Starting tuning process, please wait..." << endl;

    Order tuneOrder = getWork(true);
    QString tuneCmdLine("./leelaz --batchsize=5 --tune-only -w networks/");
    tuneCmdLine.append(tuneOrder.parameters()["network"] + ".gz");
    if (m_gpusList.isEmpty()) {
        runTuningProcess(tuneCmdLine);
    } else {
        for (auto i = 0; i < m_gpusList.size(); ++i) {
            runTuningProcess(tuneCmdLine + " --gpu=" + m_gpusList.at(i));
        }
    }
    QTextStream(stdout) << "Tuning process finished" << endl;

    m_start = std::chrono::high_resolution_clock::now();
    QString myGpu;
    for (int gpu = 0; gpu < m_gpus; ++gpu) {
        for (int game = 0; game < m_games; ++game) {
            int thread_index = gpu * m_games + game;
            if (m_gpusList.isEmpty()) {
                myGpu = "";
            } else {
                myGpu = m_gpusList.at(gpu);
            }
            QTextStream(stdout) << "Starting thread " << game + 1 ;
            QTextStream(stdout) << " on device " << gpu << endl;
            m_gamesThreads[thread_index] = new Worker(thread_index, myGpu, this);
            connect(m_gamesThreads[thread_index],
                    &Worker::resultReady,
                    this,
                    &Management::getResult,
                    Qt::DirectConnection);
            QFileInfo finfo = getNextStored();
            if (!finfo.fileName().isEmpty()) {
                m_gamesThreads[thread_index]->order(getWork(finfo));
            } else {
                m_gamesThreads[thread_index]->order(getWork());
            }
            m_gamesThreads[thread_index]->start();
        }
    }
}

void Management::storeGames() {
    for (int i = 0; i < m_gpus * m_games; ++i) {
        m_gamesThreads[i]->doStore();
    }
    wait();
}

void Management::wait() {
    QTextStream(stdout) << "Management: waiting for workers" << endl;
    for (int i = 0; i < m_gpus * m_games; ++i) {
        m_gamesThreads[i]->wait();
        QTextStream(stdout) << "Management: Worker " << i+1 << " ended" << endl;
    }
}

void Management::getResult(Order ord, Result res, int index, int duration) {
    if (res.type() == Result::Error) {
        exit(1);
    }
    m_syncMutex.lock();
    m_gamesPlayed++;
    switch (res.type()) {
    case Result::File:
        m_selfGames++,
        uploadData(res.parameters(), ord.parameters());
        printTimingInfo(duration);
        break;
    case Result::Win:
    case Result::Loss:
        m_matchGames++,
        uploadResult(res.parameters(), ord.parameters());
        printTimingInfo(duration);
        break;
    }
    sendAllGames();
    if (m_gamesLeft == 0) {
        m_gamesThreads[index]->doFinish();
        if (m_threadsLeft > 1) {
            --m_threadsLeft;
        } else {
            sendQuit();
        }
    } else {
        if (m_gamesLeft > 0) --m_gamesLeft;
        QFileInfo finfo = getNextStored();
        if (!finfo.fileName().isEmpty()) {
            m_gamesThreads[index]->order(getWork(finfo));
        } else {
            m_gamesThreads[index]->order(getWork());
        }
    }
    m_syncMutex.unlock();
}

QFileInfo Management::getNextStored() {
    QFileInfo fi;
    checkStoredGames();
    while (!m_storedFiles.isEmpty()) {
        fi = m_storedFiles.takeFirst();
        m_lockFile = new QLockFile(fi.fileName()+".lock");
        if (m_lockFile->tryLock(10) &&
           fi.exists()) {
                break;
        }
        delete m_lockFile;
        m_lockFile = nullptr;
    }
    return fi;
}

void  Management::printTimingInfo(float duration) {

    auto game_end = std::chrono::high_resolution_clock::now();
    auto total_time_s =
        std::chrono::duration_cast<std::chrono::seconds>(game_end - m_start);
    auto total_time_min =
        std::chrono::duration_cast<std::chrono::minutes>(total_time_s);
    auto total_time_millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_time_s);
    QTextStream(stdout)
        << m_gamesPlayed << " game(s) (" << m_selfGames << " self played and "
        << m_matchGames << " matches) played in "
        << total_time_min.count() << " minutes = "
        << total_time_s.count() / m_gamesPlayed << " seconds/game, "
        << total_time_millis.count() / m_movesMade.load()  << " ms/move"
        << ", last game took " << int(duration) << " seconds." << endl;
}

QString Management::getOption(const QJsonObject &ob, const QString &key, const QString &opt, const QString &defValue) {
    QString res;
    if (ob.contains(key)) {
        res.append(opt + ob.value(key).toString() + " ");
    } else {
        if (defValue != "") {
            res.append(opt + defValue + " ");
        }
    }
    return res;
}

QString Management::getBoolOption(const QJsonObject &ob, const QString &key, const QString &opt, bool defValue) {
    QString res;
    if (ob.contains(key)) {
        if (ob.value(key).toString().compare("true", Qt::CaseInsensitive) == 0) {
            res.append(opt + " ");
        }
    } else {
        if (defValue) {
            res.append(opt + " ");
        }
    }
    return res;
}

QString Management::getOptionsString(const QJsonObject &opt, const QString &rnd) {
    QString options;
    options.append(getOption(opt, "playouts", " -p ", ""));
    options.append(getOption(opt, "visits", " -v ", ""));
    options.append(getOption(opt, "resignation_percent", " -r ", "1"));
    options.append(getOption(opt, "randomcnt", " -m ", "30"));
    options.append(getOption(opt, "threads", " -t ", "6"));
    options.append(getOption(opt, "batchsize", " --batchsize ", "5"));
    options.append(getBoolOption(opt, "dumbpass", " -d ", true));
    options.append(getBoolOption(opt, "noise", " -n ", true));
    options.append(" --noponder ");
    if (rnd != "") {
        options.append(" -s " + rnd + " ");
    }
    return options;
}

QString Management::getGtpCommandsString(const QJsonValue &gtpCommands) {
    const auto gtpCommandsJsonDoc = QJsonDocument(gtpCommands.toArray());
    const auto gtpCommandsJson = gtpCommandsJsonDoc.toJson(QJsonDocument::Compact);
    auto gtpCommandsString = QVariant(gtpCommandsJson).toString();
    gtpCommandsString.remove(QRegularExpression("[\\[\\]\"]"));
    return gtpCommandsString;
}

Order Management::getWorkInternal(bool tuning) {
    Order o(Order::Error);

    /*

{
   cmd : "match",
   white_hash : "223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6",
   black_hash : "92c658d7325fe38f0c8adbbb1444ed17afd891b9f208003c272547a7bcb87909",
   options_hash : "c2e3",
   minimum_autogtp_version: "16",
   random_seed: "2301343010299460478",
   minimum_leelaz_version: "0.15",
   options : {
       playouts : "1000",
       visits: "3201",
       resignation_percent : "3",
       noise : "true",
       randomcnt : "30"
    },
    white_options : {
       playouts : "0",
       visits: "1601",
       resignation_percent : "5",
       noise : "false",
       randomcnt : "0"
    },
    white_hash_gzip_hash: "23c29bf777e446b5c3fb0e6e7fa4d53f15b99cc0c25798b70b57877b55bf1638",
    black_hash_gzip_hash: "ccfe6023456aaaa423c29bf777e4aab481245289aaaabb70b7b5380992377aa8",
    hash_sgf_hash: "7dbccc5ad9eb38f0135ff7ec860f0e81157f47dfc0a8375cef6bf1119859e537",
    moves_count: "92",
    gtp_commands : [ "time_settings 600 30 1", "komi 0.5", "fixed_handicap 2" ],
    white_gtp_commands : [ "time_settings 0 10 1", "komi 0.5", "fixed_handicap 2" ],
}

{
   cmd : "selfplay",
   hash : "223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6",
   options_hash : "ee21",
   minimum_autogtp_version: "16",
   random_seed: "2301343010299460478",
   minimum_leelaz_version: "0.15",
   options : {
       playouts : "1000",
       visits: "3201",
       resignation_percent : "3",
       noise : "true",
       randomcnt : "30"
    },
    hash_gzip_hash: "23c29bf777e446b5c3fb0e6e7fa4d53f15b99cc0c25798b70b57877b55bf1638",
    hash_sgf_hash: "7dbccc5ad9eb38f0135ff7ec860f0e81157f47dfc0a8375cef6bf1119859e537",
    moves_count: "92",
    gtp_commands : [ "time_settings 600 30 1", "komi 0.5", "fixed_handicap 4" ],
}

{
   "cmd" : "wait",
   "minutes" : "5",
}

    */
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    prog_cmdline.append(" -s -J");
    prog_cmdline.append(" "+server_url+"get-task/");
    if (tuning) {
        prog_cmdline.append("0");
    } else {
        prog_cmdline.append(QString::number(AUTOGTP_VERSION));
        if (!m_leelaversion.isEmpty())
            prog_cmdline.append("/"+m_leelaversion);
    }
    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    if (curl.exitCode()) {
        throw NetworkException("Curl returned non-zero exit code "
                               + std::to_string(curl.exitCode()));
        return o;
    }
    QJsonDocument doc;
    QJsonParseError parseError;
    doc = QJsonDocument::fromJson(curl.readAllStandardOutput(), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        std::string errorString = parseError.errorString().toUtf8().constData();
        throw NetworkException("JSON parse error: " + errorString);
    }

    if (!tuning) {
        QTextStream(stdout) << doc.toJson() << endl;
    }
    QMap<QString,QString> parameters;
    QJsonObject ob = doc.object();
    //checking client version
    int required_version = 0;
    if (ob.contains("required_client_version")) {
        required_version = ob.value("required_client_version").toString().toInt();
    } else if (ob.contains("minimum_autogtp_version")) {
        required_version = ob.value("minimum_autogtp_version").toString().toInt();
    }
    if (required_version > m_version) {
        QTextStream(stdout) << "Required client version: " << required_version << endl;
        QTextStream(stdout) << ' ' <<  endl;
        QTextStream(stdout)
            << "Server requires client version " << required_version
            << " but we are version " << m_version << endl;
        QTextStream(stdout)
            << "Check https://github.com/gcp/leela-zero for updates." << endl;
        exit(EXIT_FAILURE);
    }
    //passing leela version
    QString leelazVersion = Leelaz_min_version;
    if (ob.contains("leelaz_version")) {
        leelazVersion = ob.value("leelaz_version").toString();
    } else if (ob.contains("minimum_leelaz_version")) {
        leelazVersion = ob.value("minimum_leelaz_version").toString();
    }
    parameters["leelazVer"] = leelazVersion;

    //getting the random seed
    QString rndSeed = "0";
    if (ob.contains("random_seed")) {
        rndSeed = ob.value("random_seed").toString();
    }
    parameters["rndSeed"] = rndSeed;
    if (rndSeed == "0") {
        rndSeed = "";
    }

    //parsing options
    if (ob.contains("options")) {
        parameters["optHash"] = ob.value("options_hash").toString();
        parameters["options"] = getOptionsString(ob.value("options").toObject(), rndSeed);
    }
    if (ob.contains("gtp_commands")) {
        parameters["gtpCommands"] = getGtpCommandsString(ob.value("gtp_commands"));
    }
    if (ob.contains("hash_sgf_hash")) {
        parameters["sgf"] = fetchGameData(ob.value("hash_sgf_hash").toString(), "sgf");
        parameters["moves"] = ob.contains("moves_count") ?
            ob.value("moves_count").toString() : "0";
    }

    parameters["debug"] = !m_debugPath.isEmpty() ? "true" : "false";

    if (!tuning) {
        QTextStream(stdout) << "Got new job: " << ob.value("cmd").toString() << endl;
    }
    if (ob.value("cmd").toString() == "selfplay") {
        QString net = ob.value("hash").toString();
        QString gzipHash = ob.value("hash_gzip_hash").toString();
        fetchNetwork(net, gzipHash);
        parameters["network"] = net;

        o.type(Order::Production);
        o.parameters(parameters);
        if (m_delNetworks &&
            m_fallBack.parameters()["network"] != net) {
            QTextStream(stdout) << "Deleting network " << "networks/"
                + m_fallBack.parameters()["network"] + ".gz" << endl;
            QFile::remove("networks/" + m_fallBack.parameters()["network"] + ".gz");
        }
        m_fallBack = o;
        QTextStream(stdout) << "net: " << net << "." << endl;
    }
    if (ob.value("cmd").toString() == "match") {
        QString net1 = ob.value("black_hash").toString();
        QString gzipHash1 = ob.value("black_hash_gzip_hash").toString();
        QString net2 = ob.value("white_hash").toString();
        QString gzipHash2 = ob.value("white_hash_gzip_hash").toString();
        fetchNetwork(net1, gzipHash1);
        fetchNetwork(net2, gzipHash2);
        parameters["firstNet"] = net1;
        parameters["secondNet"] = net2;
        parameters["optionsSecond"] = ob.contains("white_options") ?
            getOptionsString(ob.value("white_options").toObject(), rndSeed) :
            parameters["options"];
        if (ob.contains("gtp_commands")) {
            parameters["gtpCommandsSecond"] = ob.contains("white_gtp_commands") ?
                getGtpCommandsString(ob.value("white_gtp_commands")) :
                parameters["gtpCommands"];
        }

        o.type(Order::Validation);
        o.parameters(parameters);
        if (m_delNetworks) {
            if (m_lastMatch.parameters()["firstNet"] != net1 &&
                m_lastMatch.parameters()["firstNet"] != net2) {
                QTextStream(stdout) << "Deleting network " << "networks/"
                    + m_lastMatch.parameters()["firstNet"] + ".gz" << endl;
                QFile::remove("networks/" + m_lastMatch.parameters()["firstNet"] + ".gz");
            }
            if (m_lastMatch.parameters()["secondNet"] != net1 &&
                m_lastMatch.parameters()["secondNet"] != net2) {
                QTextStream(stdout) << "Deleting network " << "networks/"
                    + m_lastMatch.parameters()["secondNet"] + ".gz" << endl;
                QFile::remove("networks/" + m_lastMatch.parameters()["secondNet"] + ".gz");
            }
        }
        m_lastMatch = o;
        QTextStream(stdout) << "first network: " << net1 << "." << endl;
        QTextStream(stdout) << "second network " << net2 << "." << endl;
    }
    if (ob.value("cmd").toString() == "wait") {
        parameters["minutes"] = ob.value("minutes").toString();

        o.type(Order::Wait);
        o.parameters(parameters);
        QTextStream(stdout) << "minutes: " << parameters["minutes"]  << "." << endl;
    }
    return o;
}

Order Management::getWork(bool tuning) {
    for (auto retries = 0; retries < MAX_RETRIES; retries++) {
        try {
            return getWorkInternal(tuning);
        } catch (const NetworkException &ex) {
            QTextStream(stdout)
                << "Network connection to server failed." << endl;
            QTextStream(stdout)
                << ex.what() << endl;
            auto retry_delay =
                std::min<int>(
                    RETRY_DELAY_MIN_SEC * std::pow(1.5, retries),
                    RETRY_DELAY_MAX_SEC);
            QTextStream(stdout) << "Retrying in " << retry_delay << " s."
                                << endl;
            QThread::sleep(retry_delay);
        }
    }
    QTextStream(stdout) << "Maximum number of retries exceeded. Falling back to previous network."
                        << endl;
    if (m_fallBack.type() != Order::Error) {
        QMap<QString,QString> map = m_fallBack.parameters();
        QString seed = QString::number(QUuid::createUuid().toRfc4122().toHex().left(8).toLongLong(Q_NULLPTR, 16));
        QString rs = "-s " + seed + " ";
        map["rndSeed"] = seed;
        QString opt = map["options"];
        QRegularExpression re("-s .* ");
        opt.replace(re, rs);
        map["options"] = opt;
        m_fallBack.parameters(map);
        return m_fallBack;
    }
    exit(EXIT_FAILURE);
}


bool Management::networkExists(const QString &name, const QString &gzipHash) {
    if (QFileInfo::exists(name)) {
        QFile f(name);
        if (f.open(QFile::ReadOnly)) {
            QCryptographicHash hash(QCryptographicHash::Sha256);
            if (!hash.addData(&f)) {
                throw NetworkException("Reading network file failed.");
            }
            QString result = hash.result().toHex();
            if (result == gzipHash) {
                return true;
            }
            QTextStream(stdout) << "Downloaded network hash doesn't match, calculated: "
                << result << " it should be: " << gzipHash << endl;
        } else {
            QTextStream(stdout)
                << "Unable to open network file for reading." << endl;
            if (f.remove()) {
                return false;
            }
            throw NetworkException("Unable to delete the network file."
                                   " Check permissions.");
        }
    }
    return false;
}

void Management::fetchNetwork(const QString &net, const QString &hash) {
    QString name = "networks/" + net + ".gz";
    if (networkExists(name, hash)) {
        return;
    }
    if (QFileInfo::exists(name)) {
        QFile f_gz(name);
        // Curl refuses to overwrite, so make sure to delete the gzipped
        // network if it exists
        f_gz.remove();
    }

    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    // Be quiet, but output the real file name we saved.
    // Use the filename from the server.
    prog_cmdline.append(" -s -J -o " + name + " ");
    prog_cmdline.append(" -w %{filename_effective}");
    prog_cmdline.append(" "+server_url + name);

    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    if (curl.exitCode()) {
        throw NetworkException("Curl returned non-zero exit code "
                               + std::to_string(curl.exitCode()));
    }

    QByteArray output = curl.readAllStandardOutput();
    QString outstr(output);
    QStringList outlst = outstr.split("\n");
    QString outfile = outlst[0];
    QTextStream(stdout) << "Net filename: " << outfile << endl;
    return;
}

QString Management::fetchGameData(const QString &name, const QString &extension) {
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif

    const auto fileName = QUuid::createUuid().toRfc4122().toHex();

    // Be quiet, but output the real file name we saved.
    // Use the filename from the server.
    prog_cmdline.append(" -s -J -o " + fileName + "." + extension);
    prog_cmdline.append(" -w %{filename_effective}");
    prog_cmdline.append(" "+server_url + "view/" + name + "." + extension);

    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);

    if (curl.exitCode()) {
        throw NetworkException("Curl returned non-zero exit code "
                               + std::to_string(curl.exitCode()));
    }

    return fileName;
}

void Management::archiveFiles(const QString &fileName) {
    if (!m_keepPath.isEmpty()) {
        QFile(fileName + ".sgf").copy(m_keepPath + '/' + fileName + ".sgf");
    }
    if (!m_debugPath.isEmpty()) {
        QFile d(fileName + ".txt.0.gz");
        if (d.exists()) {
            d.copy(m_debugPath + '/' + fileName + ".txt.0.gz");
        }
        QFile db(fileName + ".debug.txt.0.gz");
        if (db.exists()) {
            db.copy(m_debugPath + '/' + fileName + ".debug.txt.0.gz");
        }
    }
}
void Management::cleanupFiles(const QString &fileName) {
    QDir dir;
    QStringList filters;
    filters << fileName + ".*";
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    QFileInfoList list = dir.entryInfoList();
    for (int i = 0; i < list.size(); ++i) {
        QFile(list.at(i).fileName()).remove();
    }
}

void Management::gzipFile(const QString &fileName) {
    QString gzipCmd ="gzip";
#ifdef WIN32
    gzipCmd.append(".exe");
#endif
    gzipCmd.append(" " + fileName);
    QProcess::execute(gzipCmd);
}

void Management::saveCurlCmdLine(const QStringList &prog_cmdline, const QString &name) {
    QString fileName = "curl_save" + QUuid::createUuid().toRfc4122().toHex() + ".bin";
    QLockFile lf(fileName + ".lock");
    lf.lock();
    QFile f(fileName);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return;
    }
    QTextStream out(&f);
    out << name << endl;
    out << prog_cmdline.size() << endl;
    QStringList::ConstIterator it = prog_cmdline.constBegin();
    while (it != prog_cmdline.constEnd()) {
        out << *it << " " << endl;
        ++it;
    }
    f.close();
}

void Management::sendAllGames() {
    QDir dir;
    QStringList filters;
    filters << "curl_save*.bin";
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    QFileInfoList list = dir.entryInfoList();
    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fileInfo = list.at(i);
        QLockFile lf(fileInfo.fileName()+".lock");
        if (!lf.tryLock(10)) {
            continue;
        }
        QFile file(fileInfo.fileName());
        if (!file.open(QFile::ReadOnly)) {
            continue;
        }
        QTextStream in(&file);
        QString name;
        QString tmp;
        QStringList lines;
        int count;
        in >> name;
        in >> count;
        count = 2 * count - 1;
        for (int i = 0; i < count; i++) {
            in >> tmp;
            lines << tmp;
        }
        file.close();
        bool sent = false;

        try {
            sent = sendCurl(lines);
            if (sent) {
                QTextStream(stdout) << "File: " << file.fileName() << " sent" << endl;
                file.remove();
                cleanupFiles(name);
                if (i+1 < list.size()) {
                    QThread::sleep(10);
                }
            }
        } catch (const NetworkException &ex) {
            QTextStream(stdout)
                << "Network connection to server failed." << endl;
            QTextStream(stdout)
                << ex.what() << endl;
            QTextStream(stdout)
                    << "Retrying when next game is finished."
                    << endl;
        }
    }
}

bool Management::sendCurl(const QStringList &lines) {
    QString prog_cmdline("curl");
#ifdef WIN32
    prog_cmdline.append(".exe");
#endif
    QStringList::ConstIterator it = lines.constBegin();
    while (it != lines.constEnd()) {
        prog_cmdline.append(" " + *it);
        ++it;
    }
    QProcess curl;
    curl.start(prog_cmdline);
    curl.waitForFinished(-1);
    if (curl.exitCode()) {
        QTextStream(stdout) << "Upload failed. Curl Exit code: "
            << curl.exitCode() << endl;
        QTextStream(stdout) << curl.readAllStandardOutput();
        throw NetworkException("Curl returned non-zero exit code "
                                   + std::to_string(curl.exitCode()));
        return false;
    }
    QTextStream(stdout) << curl.readAllStandardOutput();
    return (curl.exitCode() == 0);
}

/*
-F winnerhash=223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6
-F loserhash=92c658d7325fe38f0c8adbbb1444ed17afd891b9f208003c272547a7bcb87909
-F clientversion=6
-F winnercolor=black
-F movescount=321
-F score=B+45
-F options_hash=c2e3
-F random_seed=0
-F sgf=@file
https://zero.sjeng.org/submit-match
*/

void Management::uploadResult(const QMap<QString,QString> &r, const QMap<QString,QString> &l) {
    QTextStream(stdout) << "Uploading match: " << r["file"] << ".sgf for networks ";
    QTextStream(stdout) << l["firstNet"] << " and " << l["secondNet"] << endl;
    archiveFiles(r["file"]);
    gzipFile(r["file"] + ".sgf");
    QStringList prog_cmdline;
    if (r["winner"] == "black") {
        prog_cmdline.append("-F winnerhash=" + l["firstNet"]);
        prog_cmdline.append("-F loserhash=" + l["secondNet"]);
    } else {
        prog_cmdline.append("-F winnerhash=" + l["secondNet"]);
        prog_cmdline.append("-F loserhash=" + l["firstNet"]);
    }
    prog_cmdline.append("-F clientversion=" + QString::number(m_version));
    prog_cmdline.append("-F winnercolor="+ r["winner"]);
    prog_cmdline.append("-F movescount="+ r["moves"]);
    prog_cmdline.append("-F score="+ r["score"]);
    prog_cmdline.append("-F options_hash="+ l["optHash"]);
    prog_cmdline.append("-F random_seed="+ l["rndSeed"]);
    prog_cmdline.append("-F sgf=@"+ r["file"] + ".sgf.gz");
    prog_cmdline.append(server_url+"submit-match");

    bool sent = false;
    for (auto retries = 0; retries < MAX_RETRIES; retries++) {
        try {
            sent = sendCurl(prog_cmdline);
            break;
        } catch (const NetworkException &ex) {
            QTextStream(stdout)
                << "Network connection to server failed." << endl;
            QTextStream(stdout)
                << ex.what() << endl;
            auto retry_delay =
                std::min<int>(
                    RETRY_DELAY_MIN_SEC * std::pow(1.5, retries),
                    RETRY_DELAY_MAX_SEC);
            QTextStream(stdout) << "Retrying in " << retry_delay << " s."
                                << endl;
            QThread::sleep(retry_delay);
        }
    }
    if (!sent) {
        saveCurlCmdLine(prog_cmdline, r["file"]);
        return;
    }
    cleanupFiles(r["file"]);
}


/*
-F networkhash=223737476718d58a4a5b0f317a1eeeb4b38f0c06af5ab65cb9d76d68d9abadb6
-F clientversion=6
-F options_hash=ee21
-F random_seed=1
-F sgf=@file
-F trainingdata=@data_file
https://zero.sjeng.org/submit
*/

void Management::uploadData(const QMap<QString,QString> &r, const QMap<QString,QString> &l) {
    QTextStream(stdout) << "Uploading game: " << r["file"] << ".sgf for network " << l["network"] << endl;
    archiveFiles(r["file"]);
    gzipFile(r["file"] + ".sgf");
    QStringList prog_cmdline;
    prog_cmdline.append("-F networkhash=" + l["network"]);
    prog_cmdline.append("-F clientversion=" + QString::number(m_version));
    prog_cmdline.append("-F options_hash="+ l["optHash"]);
    prog_cmdline.append("-F movescount="+ r["moves"]);
    prog_cmdline.append("-F winnercolor="+ r["winner"]);
    prog_cmdline.append("-F random_seed="+ l["rndSeed"]);
    prog_cmdline.append("-F sgf=@" + r["file"] + ".sgf.gz");
    prog_cmdline.append("-F trainingdata=@" + r["file"] + ".txt.0.gz");
    prog_cmdline.append(server_url+"submit");

    bool sent = false;
    for (auto retries = 0; retries < MAX_RETRIES; retries++) {
        try {
            sent = sendCurl(prog_cmdline);
            break;
        } catch (const NetworkException &ex) {
            QTextStream(stdout)
                << "Network connection to server failed." << endl;
            QTextStream(stdout)
                << ex.what() << endl;
            auto retry_delay =
                std::min<int>(
                    RETRY_DELAY_MIN_SEC * std::pow(1.5, retries),
                    RETRY_DELAY_MAX_SEC);
            QTextStream(stdout) << "Retrying in " << retry_delay << " s."
                                << endl;
            QThread::sleep(retry_delay);
        }
    }
    if (!sent) {
        saveCurlCmdLine(prog_cmdline, r["file"]);
        return;
    }
    cleanupFiles(r["file"]);
}

void Management::checkStoredGames() {
    QDir dir;
    QStringList filters;
    filters << "storefile*.bin";
    dir.setNameFilters(filters);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    m_storedFiles = dir.entryInfoList();
}
