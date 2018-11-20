#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2018 Michael O
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import random
import unittest

class ShuffleBuffer:
    def __init__(self, elem_size, elem_count):
        """
            A shuffle buffer for fixed sized elements.

            Manages 'elem_count' items in a fixed buffer, each item being exactly
            'elem_size' bytes.
        """
        assert elem_size > 0, elem_size
        assert elem_count > 0, elem_count
        # Size of each element.
        self.elem_size = elem_size
        # Number of elements in the buffer.
        self.elem_count = elem_count
        # Fixed size buffer used to hold all the element.
        self.buffer = bytearray(elem_size * elem_count)
        # Number of elements actually contained in the buffer.
        self.used = 0

    def extract(self):
        """
            Return an item from the shuffle buffer.

            If the buffer is empty, returns None
        """
        if self.used < 1:
            return None
        # The items in the shuffle buffer are held in shuffled order
        # so returning the last item is sufficient.
        self.used -= 1
        i = self.used
        return self.buffer[i * self.elem_size : (i+1) * self.elem_size]

    def insert_or_replace(self, item):
        """
            Inserts 'item' into the shuffle buffer, returning
            a random item.

            If the buffer is not yet full, returns None
        """
        assert len(item) == self.elem_size, len(item)
        # putting the new item in a random location, and appending
        # the displaced item to the end of the buffer achieves a full
        # random shuffle (Fisher-Yates)
        if self.used > 0:
            # swap 'item' with random item in buffer.
            i = random.randint(0, self.used-1)
            old_item = self.buffer[i * self.elem_size : (i+1) * self.elem_size]
            self.buffer[i * self.elem_size : (i+1) * self.elem_size] = item
            item = old_item
        # If the buffer isn't yet full, append 'item' to the end of the buffer.
        if self.used < self.elem_count:
            # Not yet full, so place the returned item at the end of the buffer.
            i = self.used
            self.buffer[i * self.elem_size : (i+1) * self.elem_size] = item
            self.used += 1
            return None
        return item
        
 
class ShuffleBufferTest(unittest.TestCase):
    def test_extract(self):
        sb = ShuffleBuffer(3, 1)
        r = sb.extract()
        assert r == None, r # empty buffer => None
        r = sb.insert_or_replace(b'111')
        assert r == None, r # buffer not yet full => None
        r = sb.extract()
        assert r == b'111', r # one item in buffer => item
        r = sb.extract()
        assert r == None, r # buffer empty => None
    def test_wrong_size(self):
        sb = ShuffleBuffer(3, 1)
        try:
            sb.insert_or_replace(b'1') # wrong length, so should throw.
            assert False # Should not be reached.
        except:
            pass
    def test_insert_or_replace(self):
        n=10 # number of test items.
        items=[bytes([x,x,x]) for x in range(n)]
        sb = ShuffleBuffer(elem_size=3, elem_count=2)
        out=[]
        for i in items:
            r = sb.insert_or_replace(i)
            if not r is None:
                out.append(r)
        # Buffer size is 2, 10 items, should be 8 seen so far.
        assert len(out) == n - 2, len(out)
        # Get the last two items.
        out.append(sb.extract())
        out.append(sb.extract())
        assert sorted(items) == sorted(out), (items, out)
        # Check that buffer is empty
        r = sb.extract()
        assert r is None, r


if __name__ == '__main__':
    unittest.main()
