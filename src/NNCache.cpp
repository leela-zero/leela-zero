/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Michael O

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

#include "NNCache.h"

#include <functional>

#include "Utils.h"

NNCache::NNCache(int size) : m_size(size) {}

NNCache* NNCache::get_NNCache(void) {
  static NNCache* cache = new NNCache();
  return cache;
}

template <class T>
inline size_t hash_combine(size_t seed, const T& v)
{
      std::hash<T> hasher;
      return seed ^ (hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2));
}

static size_t compute_hash(const Network::NNPlanes& features) {
    size_t hash = 0;
    for (const auto& p : features) {
        hash = hash_combine(hash, p);
    }
    return hash;
}

const Network::Netresult* NNCache::lookup(const Network::NNPlanes& features) {
  LOCK(m_mutex, lock);
  ++m_lookups;

  size_t hash = compute_hash(features);
  auto iter = m_cache.find(hash);
  if (iter == m_cache.end()) return nullptr;  // Not found.

  const auto& entry = iter->second;
  if (entry->features != features) {
      // Got a hash collision.
      ++m_collisions;
      return nullptr;
  }

  // Found it.
  ++m_hits;
  return &entry->result;
}

void NNCache::insert(const Network::NNPlanes& features, const Network::Netresult& result) {
  LOCK(m_mutex, lock);

  size_t hash = compute_hash(features);
  if (m_cache.count(hash)) return;  // Already in the cache.

  m_cache.emplace(hash, std::make_unique<Entry>(features, result));
  m_order.push_back(hash);
  ++m_inserts;


  // If the cache is too large, remove the oldest entry.
  if (m_order.size() > m_size) {
      m_cache.erase(m_order.front());
      m_order.pop_front();
  }

}

void NNCache::dump_stats() {
    Utils::myprintf("NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %d size, %d collisions\n",
        m_hits, m_lookups, 100. * m_hits / (m_lookups + 1), m_inserts, m_cache.size(), m_collisions);
}
