#!/usr/bin/env python3
# Format for v3 is as follows 

# 5 bytes Magic Number '3LZW\n'
# 1 byte for value head type: 0 for v1 type, 1 for v2 type
# 1 byte float size: 0 for 16-bit, 1 for 32-bit
# 2 bytes number of residual blocks (unsigned integer)
# 2 bytes number of filters (unsigned integer)
# From here, the order of numbers is exactly the same as in the v1 file,
# directly in IEEE 754-2008 little endian format.

# Data sanity:
# Floating point numbers MUST NOT encode a non-finite number
# Size of number of residual blocks and filters must be non-zero

import sys
import struct
import getopt
import numpy

# The newline is to minimize the changes to the engine, and keep the detection
# method unified.
magic = str.encode('3LZW\n')

# The conversion functions *can* give a non-finite value in certain pathological
# cases; however, good networks should never trigger them, so bounds checking is
# ignored for now.

# Takes a python float and returns a 16-bit byte array of that float's
# representation
def conv16(number):
    f16 = numpy.float16(number)
    out = bytes(f16)
    if sys.byteorder == 'big':
        return out[::-1]
    else:
        return out

# Takes a python float and returns a 32-bit byte array of that float's
# representation
def conv32(number):
    return struct.pack("<f", number)

def process(float_size, f_in, f_out, elf):
    float_byte = 0
    if (float_size == 16):
        float_byte = 0
    elif (float_size == 32):
        float_byte = 1

    # Unfortunately, in order to determine the number of blocks and filters, we
    # must scan the entire file, it's entirely possible to do this without
    # reading it all in, and while this does use a lot of memory, it's not
    # *that* big of a problem, and a second pass can fix this
    lines = f_in.readlines()

    # 1 format id, 1 input layer (4 x weights), 14 ending weights,
    # the rest are residuals, every residual has 8 x weight lines
    blocks = len(lines) - (1 + 4 + 14)
    if blocks % 8 != 0:
        print("Inconsistent number of weights in the file", file=sys.stderr)
        return
    blocks //= 8

    if (blocks > 65535):
        print("Too many blocks (%d) for this file format (also wow).", blocks,
        file=sys.stderr)
        sys.exit(-2)
    
    filters = len([float(i) for i in lines[2].split(" ")])
    
    if (filters > 65535):
        print("Too many filters (%d) for this file format (also wow).", filters,
        file=sys.stderr)
        sys.exit(-2)

    print("Found %d blocks and %d filters" % (blocks, filters), file=sys.stderr)

    # And now let's write out all of this out to the output
    f_out.write(magic)
    if (elf):
        f_out.write(struct.pack('B', 1))
    else:
        f_out.write(struct.pack('B', 0))
    f_out.write(struct.pack('B', float_byte))
    f_out.write(struct.pack('<H', blocks))
    f_out.write(struct.pack('<H', filters))

    # Fetch the correct conversion function
    conv = [conv16, conv32][float_byte]

    # And finally dump all of the weights
    for line in lines[1:]: # Skip first line because it's version information
        for weight in [float(i) for i in line.split(" ")]: # or bias
            f_out.write(conv(weight))

def print_usage():
    print(
"""Usage: %s [OPTION]...
Convert a Leela Zero weights file from the v1 format to the v3 format

With no input specified using the -i option, read standard input
With no output specified using the -o option, write standard output

  -s, --float_size\tNumber of bits per float, default 16,
\t\t\toptions are 16, 32
  -i, --input\t\tName of file from which to read weights from
  -o, --output\t\tName of file to which to write weights to
  -e, --elf_weights\tFlags the weights as using the ELF
\t\t\tformat for value head, default off
""" % sys.argv[0], file=sys.stderr)

if __name__ == "__main__":
    options, remainder = getopt.getopt(sys.argv[1:], 's:i:o:e', ['float_size=',
        'input=', 'output=', 'elf_weights'])

    if len(remainder):
        print("Unrecognized command line options (%s)" % remainder, file=sys.stderr)
        print_usage()
        sys.exit(-1)

    float_size = 16
    input_file = ''
    output_file = ''
    elf_weights = False

    for opt, arg in options:
        if opt in ['-s', '--float_size']:
            if int(arg) in [16, 32]:
                float_size = int(arg)
            else:
                print("Invalid size specified (%s), units are bits and options"
                " are 16, 32" % arg, file=sys.stderr)
                print_usage()
                sys.exit(-1)
        elif opt in ['-i', '--input']:
            input_file = arg
        elif opt in ['-o', '--output']:
            output_file = arg
        elif opt in ['-e', '--elf_weights']:
            elf_weights = True
        else:
            print("Invalid option specified (%s)" % opt, file=sys.stderr)
            print_usage()
            sys.exit(-1)

    input_handle = []
    output_handle = []
    if len(input_file):
        input_handle = open(input_file)
    else:
        input_handle = sys.stdin

    if len(output_file):
        output_handle = open(output_file, 'wb')
    else:
        output_handle = sys.stdout.buffer

    process(float_size, input_handle, output_handle, elf_weights)
    input_handle.close()
    output_handle.close()
