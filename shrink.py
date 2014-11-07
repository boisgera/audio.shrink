#!/usr/bin/env python
# coding: utf-8

"""
SHRINK Lossless Audio Coding
"""

# Python Standard Library
import argparse
import doctest
import inspect
import os
import sys

# Third-Party Libraries
from numpy import *
from numpy.linalg import *

# Digital Audio Coding Library
from bitstream import BitStream
import breakpoint
from coders import rice
from frames import split
import wave
import script
import logger # TODO: convert to logfile, remove tagging, error hooks, etc.

#
# Notes, TODO, etc.
# ------------------------------------------------------------------------------
#


# **BUG:** Issue with the whole code that expects all predictors never to overflow. 
# As soon as it happens, diff then cumsum is not the identity anymore (!).
# In the low-order predictors, the problem could be handled by making all
# computation in int32 instead of int16 for example, but the worst-case 
# for high-order prediction would probably require unbounded integers 
# (with their specific problems ...).

# A reasonable way to do it would be to check overflow from int32 or int64
# computations and to stop the increase in the prediction order as soon as
# it happens, even if seems to reduce the "size" of the prediction residual.
# The simplest way to do it is probably to invert (one step of) the delta
# and see if it is equal to the original signal.



#
# Metadata
# ------------------------------------------------------------------------------
#
__author__ = u"Sébastien Boisgérault <Sebastien.Boisgerault@mines-paristech.fr>"
__license__ = "MIT License"
__version__ = None

#
# Codecs Registration
# ------------------------------------------------------------------------------
#
class Struct(object):
    def __init__(self, **kwargs):
        "Create a structure whose content is given by the keywords arguments" 
        self.__dict__.update(kwargs)

_codecs = {}

def register(id, name, coder, decoder, doc=None):
    """
    Register a SHRINK codec pair by id (number) and name.
    """
    info = Struct(id=id, name=name, coder=coder, decoder=decoder, doc=doc)
    _codecs[id] = _codecs[name] = info

#
# Breakpoint Callback
# ------------------------------------------------------------------------------
#
def on_breakpoint(progress, elapsed, remain):
    logger.info("time remaining: {remain:.1f} secs.")

#
# BitStream Helpers
# ------------------------------------------------------------------------------
#

# TODO: use ndmin argument to array constructor instead.
def normalize(channels):
    channels = array(channels, copy=False)
    if len(shape(channels)) == 1:
        channels = reshape(channels, (1,-1))
    return channels

def zero_pad(stream):
    extra = len(stream) % 8
    if extra:
        stream.write((8 - extra) * [False])

#
# Version 0 - Amplitude Rice Coding
# ------------------------------------------------------------------------------
#
@logger.tag("shrink.coder.v0")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def shrink_v0(channels):
    channels = normalize(channels)
    stream = BitStream()
    stream.write("SHRINK")
    version = 0
    stream.write(version, uint8)
    length = shape(channels)[1]
    stream.write(length, uint32)
    stereo = shape(channels)[0] == 2
    stream.write(stereo)
    for i_channel, channel in enumerate(channels):
        if channel.size:
            n = rice.select_parameter(mean(abs(channel)))
            stream.write(n, uint8)
            codec = rice(n, signed=True)
            i = 0
            count, stop = 0, 1
            while i < length:
                i_next = min(i + 4410, length)
                stream.write(channel[i:i_next], codec)
                i = i_next
                if count >= stop:
                    count = 0
                    progress = float(i_channel + float(i) / length) / len(channels)  
                    stop = stop * (yield progress)
                count += 1
    zero_pad(stream)
    yield (1.0, stream)

@logger.tag("shrink.decoder.v0")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def grow_v0(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(uint8) == 0
    length = stream.read(uint32)
    num_channels = stream.read(bool) + 1
    channels = zeros((num_channels, length), dtype=int16)
    if channels.size:        
        for i_channel in range(num_channels):
            n = stream.read(uint8)
            codec = rice(n, signed=True)
            i = 0
            count, stop = 0, 1
            while i < length:
                i_next = min(i + 4410, length)
                channels[i_channel][i:i_next] = stream.read(codec, i_next - i)
                i = i_next
                if count >= stop:
                    count = 0
                    progress = float(i_channel + float(i) / length) / len(channels)  
                    stop = stop * (yield progress)
                count += 1
    assert all(r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(0, "v0", shrink_v0, grow_v0, "amplitude rice coder")

#
# Version 1 - Differential Rice Coding
# ------------------------------------------------------------------------------
#
@logger.tag("shrink.coder.v1")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def shrink_v1(channels):
    channels = normalize(channels)
    stream = BitStream()
    stream.write("SHRINK")
    version = 1
    stream.write(version, uint8)
    length = shape(channels)[1]
    stream.write(length, uint32)
    stereo = shape(channels)[0] == 2
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if channel.size:
            delta = diff(r_[0, channel])
            n = rice.select_parameter(mean(abs(delta)))
            codec = rice(n, signed=True)
            stream.write(n, uint8)
            i = 0
            while i < length:
                if count >= stop:
                    count = 0
                    progress  = float(i_channel) / len(channels)
                    progress += float(i) / length / len(channels) 
                    x = (yield progress)
                    stop = stop * x
                count += 1
                i_next = min(i + 4410, length)
                stream.write(delta[i:i_next], codec)
                i = i_next
    zero_pad(stream)
    yield (1.0, stream)

@logger.tag("shrink.decoder.v1")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def grow_v1(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(uint8) == 1
    length = stream.read(uint32)
    num_channels = stream.read(bool) + 1
    channels = zeros((num_channels, length), dtype=int16)
    count, stop = 0, 1
    for i in range(num_channels):
        if channels[i].size:
            n = stream.read(uint8)
            codec = rice(n, signed=True)
            delta = zeros(length, dtype=float64)
            j = 0
            while j < length:
                if count >= stop:
                    count = 0
                    progress  = float(i) / num_channels
                    progress += float(j) / length / num_channels
                    x = (yield progress)
                    stop = stop * x
                count += 1
                j_next = min(j + 4410, length)  
                delta[j:j_next] = stream.read(codec, j_next - j)
                j = j_next
            channels[i] = cumsum(delta)
    assert all(r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(1, "v1", shrink_v1, grow_v1, "differential rice coder")

#
# Version 2 - First-Order Prediction Residual Rice Coder
# ------------------------------------------------------------------------------
#
@logger.tag("shrink.coder.v2")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def shrink_v2(channels):
    channels = normalize(channels)
    stream = BitStream()
    stream.write("SHRINK")
    version = 2
    stream.write(version, uint8)
    length = shape(channels)[1]
    stream.write(length, uint32)
    stereo = shape(channels)[0] == 2
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if channel.size:
            delta = diff(r_[0, diff(r_[0, channel])])
            n = rice.select_parameter(mean(abs(delta)))
            stream.write(n, uint8)
            codec = rice(n, signed=True)
            i = 0
            while i < length:
                if count >= stop:
                    count = 0
                    progress  = float(i_channel) / len(channels)
                    progress += float(i) / length / len(channels)
                    x = (yield progress)
                    stop = stop * x
                count += 1
                i_next = min(i + 4410, length) 
                stream.write(delta[i:i_next], codec)
                i = i_next
    zero_pad(stream)
    yield (1.0, stream)

@logger.tag("shrink.decoder.v2")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def grow_v2(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(uint8) == 2
    length = stream.read(uint32)
    num_channels = stream.read(bool) + 1
    channels = zeros((num_channels, length), dtype=int16)
    count, stop = 0, 1
    for i in range(num_channels):
        if channels[i].size:
            n = stream.read(uint8)
            codec = rice(n, signed=True)
            delta = zeros(length, dtype=float64)
            j = 0
            while j < length:
                if count >= stop:
                    count = 0
                    progress  = float(i) / num_channels
                    progress += float(j) / length / num_channels
                    x = (yield progress)
                    stop = stop * x
                count += 1
                j_next = min(j + 4410, length) 
                delta[j:j_next] = stream.read(codec, j_next - j)
                j = j_next
            channels[i] = cumsum(cumsum(delta))
    assert all(r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(2, "v2", shrink_v2, grow_v2, "1st-order pred. residual rice coder")

#
# Version 3 - Polynomial Prediction Residual Rice Coder
# ------------------------------------------------------------------------------
#
@logger.tag("shrink.coder.v3")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def shrink_v3(channels, N=14):
    channels = normalize(channels)
    stream = BitStream()
    stream.write("SHRINK")
    version = 3
    stream.write(version, uint8)
    length = shape(channels)[1]
    stream.write(length, uint32)
    num_channels = shape(channels)[0]
    stereo = num_channels == 2
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if channels.size:
            mean_ = mean(abs(channel))
            i = 0
            while i <= N:
                delta = diff(r_[0, channel])
                new_mean = mean(abs(delta))
                if new_mean >= mean_:
                    break
                else:
                    mean_ = new_mean
                    channel = delta
                    i += 1
            stream.write(i, rice(3)) # we write poly. order + 1
            # so that 0 is absolute coding, 1 simple diff, etc.
            n = rice.select_parameter(mean_)
            stream.write(n, uint8)
            codec = rice(n, signed=True)
            j = 0
            while j < length:
                if count >= stop:
                    count = 0
                    progress  = float(i_channel) / num_channels
                    progress += float(j) / length / num_channels  
                    x = (yield progress)
                    stop = x * stop
                count += 1
                j_next = min(j + 4410, length)        
                stream.write(channel[j:j_next], codec)
                j = j_next
    zero_pad(stream)
    yield (1.0, stream)

@logger.tag("shrink.decoder.v3")
@breakpoint.breakpoint(dt=1.0, handler=on_breakpoint)
def grow_v3(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(uint8) == 3
    length = stream.read(uint32)
    num_channels = stream.read(bool) + 1
    channels = zeros((num_channels, length), dtype=int16)
    count, stop = 0, 1
    for j in range(num_channels):
        if channels[j].size:
            i = stream.read(rice(3)) # i = poly. order + 1
            n = stream.read(uint8)
            codec = rice(n, signed=True)
            delta = zeros(length, dtype=float64)
            k = 0
            while k < length:
                if count >= stop:
                    count = 0
                    progress  = float(j) / num_channels
                    progress += float(k) / length / num_channels
                    x = (yield progress)
                    stop = stop * x
                count += 1
                k_next = min(k + 4410, length)
                delta[k:k_next] = stream.read(codec, k_next - k)
                k = k_next
            for _ in range(i):
                delta = cumsum(delta)
            channels[j] = delta
    assert all(r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(3, "v3", shrink_v3, grow_v3, "polynomial pred. residual rice coder")

#
# Version 4 - Polynomial Prediction Residual Rice Coder Within Frames
# ------------------------------------------------------------------------------
#
@logger.tag("shrink.coder.v4")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def shrink_v4(channels, N=14, frame_length=882): # 882 = 20 ms
    channels = normalize(channels)
    stream = BitStream()
    stream.write("SHRINK")
    version = 4
    stream.write(version, uint8)
    length = shape(channels)[1]
    stream.write(length, uint32)
    stereo = shape(channels)[0] == 2
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if channel.size:
            frames = split(channel, frame_length)
            for i_frame, frame in enumerate(frames):
                if count >= stop:
                    count = 0
                    progress  = float(i_channel) / len(channels)
                    progress += float(i_frame) / len(frames) / len(channels)
                    x = (yield progress)
                    stop = stop * x
                count += 1

                mean_ = mean(abs(frame))
                i = 0
                while i <= N:
                    delta = diff(r_[0, frame])
                    new_mean = mean(abs(delta))
                    if new_mean >= mean_:
                        break
                    else:
                        mean_ = new_mean
                        frame = delta
                    i += 1
                stream.write(i, rice(3)) # we write poly. order + 1
                # so that 0 is absolute coding, 1 simple diff, etc.
                # means.append(mean_)
                n = rice.select_parameter(mean_)
                stream.write(n, uint8)
                stream.write(frame, rice(n, signed=True))
    zero_pad(stream)
    yield (1.0, stream)

@logger.tag("shrink.decoder.v4")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def grow_v4(stream, N=14, frame_length=882):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(uint8) == 4
    length = stream.read(uint32)
    num_channels = stream.read(bool) + 1
    channels = zeros((num_channels, length), dtype=int16)
    count, stop = 0, 1
    for j in range(num_channels):
        if channels[j].size:
            channel = zeros(0, dtype=int16)
            samples_left = length
            while samples_left:
                if count >= stop:
                    count = 0
                    progress  = float(j) / num_channels
                    progress += float(length - samples_left) / length / num_channels 
                    x = (yield progress)
                    stop = stop * x
                count += 1

                current_frame_length = min(frame_length, samples_left) 
                i = stream.read(rice(3)) # i = poly. order + 1
                n = stream.read(uint8)
                delta = stream.read(rice(n, signed=True), current_frame_length)
                for _ in range(i):
                    delta = cumsum(delta)
                channel = r_[channel, delta]
                samples_left -= current_frame_length
            channels[j] = channel
    assert all(r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

doc ="polynomial pred. residual rice coder within frames"
register(4, "v4", shrink_v4, grow_v4, doc)

#
# Version 5 - Polynomial Prediction Residual Rice Coder Within Overlapping Frames
# ------------------------------------------------------------------------------
#

@logger.tag("shrink.coder.v5")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def shrink_v5(channels, N=14, frame_length=882): # 882 = 20 ms
    channels = normalize(channels)
    logger.debug("beginning coding")
    stream = BitStream()
    stream.write("SHRINK")
    version = 5
    stream.write(version, uint8)
    length = shape(channels)[1]
    stream.write(length, uint32)
    stereo = shape(channels)[0] == 2
    stream.write(stereo)
    #frame_length = min(length, frame_length)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if size(channel) > 0:
            #raise RuntimeError(frame_length, N+1)
            frames = split(r_[zeros(N+1, dtype=int16), channel], 
                           frame_length = frame_length + N + 1, 
                           overlap = N + 1)
            for i_frame, frame in enumerate(frames):
                if count >= stop:
                    count = 0
                    c_progress = float(i_channel) / len(channels)
                    f_progress = float(i_frame) / len(frames)
                    x = (yield c_progress + f_progress / len(channels))
                    stop = stop * x
                count += 1

                mean_ = mean(abs(frame[N+1:]))
                i = 0
                while i <= N:
                    delta = diff(frame)
                    new_mean = mean(abs(delta[N-i:]))
                    if new_mean >= mean_:
                        break
                    else:
                        mean_ = new_mean
                        frame = delta
                    i += 1

                stream.write(i, rice(3)) # we write poly. order + 1
                # so that 0 is absolute coding, 1 simple diff, etc.
                # means.append(mean_)
                n = rice.select_parameter(mean_)
                stream.write(n, uint8)
                stream.write(frame[N-i+1:], rice(n, signed=True))
    zero_pad(stream)
    logger.debug("ending coding")
    yield (1.0, stream)

@logger.tag("shrink.decoder.v5")
@breakpoint.breakpoint(dt=10.0, handler=on_breakpoint)
def grow_v5(stream, N=14, frame_length=882):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(uint8) == 5
    length = stream.read(uint32)
    num_channels = stream.read(bool) + 1
    channels = zeros((num_channels, length), dtype=int16)
    count, stop = 0, 1
    for j in range(num_channels):
        if size(channels[j]) > 0:
            channel = None
            samples_left = length
            while samples_left:
                if count >= stop:
                    count = 0
                    progress  = float(j) / num_channels
                    progress += float(length - samples_left) / length / num_channels
                    x = (yield progress)
                    stop = stop * x
                count += 1

                current_frame_length = min(frame_length, samples_left) 
                i = stream.read(rice(3)) # i = poly. order + 1
                if channel is None:
                    channel = zeros(i, dtype=uint16)
                    offset = i 
                n = stream.read(uint8)
                delta = stream.read(rice(n, signed=True), current_frame_length)
                delta = array(delta, dtype=int16)
                sum_offset = []
                if i > 0:
                    start = channel[-i:]
                    for k in range(i):
                       sum_offset.insert(0, start[-1])
                       start = diff(start)
                    for k in range(i):
                        delta = cumsum(delta) + sum_offset[k]
                channel = r_[channel, delta]
                samples_left -= current_frame_length
            channels[j] = channel[offset:]
    assert all(r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

doc ="polynomial pred. residual rice coder within overlapping frames"
register(5, "v5", shrink_v5, grow_v5, doc)

#
# Doctests
# ------------------------------------------------------------------------------
#
def test_round_trip():
    """
Test that shrink + grow achieves perfect reconstruction.

    TODO: I would need much longer dataset to really stress-test framed
          version of the codecs.

    >>> dataset = [ [], [[]], [0], [[0], [0]], [0, 0], [[0, 0], [0, 0]],
    ...             [-2**15, -2**15+1, -2, -1, 0, 1, 2**15 -2, 2**15 - 1], 
    ...             [[-2**15, 0, 2**15-1], [0, 1, 2]],
    ...           ]
    >>> def check_round_trip(shrink, grow):
    ...     checks =  []
    ...     for data in dataset:
    ...         checks.append( all(data == grow(shrink(data))) )
    ...     return all(checks) 

    >>> check_round_trip(shrink_v0, grow_v0)
    True
    >>> check_round_trip(shrink_v1, grow_v1)
    True
    >>> check_round_trip(shrink_v2, grow_v2)
    True
    >>> check_round_trip(shrink_v3, grow_v3)
    True
    >>> check_round_trip(shrink_v4, grow_v4)
    True
    >>> check_round_trip(shrink_v5, grow_v5)
    True
"""

def test(verbose=False):
    failed, _ = doctest.testmod(verbose=verbose)
    return failed

#
# Command-Line Interface
# ------------------------------------------------------------------------------
#
def help():
    """
Return the following message:

    usage:

        shrink [OPTIONS] FILENAME

    options:

        -h / --help ................................. display help and exit
        -l / --list ................................. list available codecs
        -c CODEC / --codec=CODEC .................... select a codec (by id or name)
        -v / --verbose .............................. verbose mode (may be repeated)
        -t / --test ................................. perform self-tests
"""
    return "\n".join(line[4:] for line in inspect.getdoc(help).splitlines()[2:])

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    spec = "help list codec= test verbose"
    options, filenames = script.parse(spec, args)

    verbosity = len(options.verbose)
    logger.config.level = verbosity
    def _format(channel, message, tag):
        tag = tag or ""
        return " {0!r:<9} | {1:<18} | {2}\n".format(channel, tag, message)
    logger.config.format = _format
    def _error_hook(message, ExceptionType=ValueError):
        raise ExceptionType(message)
    logger.error.set_hook(_error_hook)

    if options.help or (not filenames and not options):
        print help()
        sys.exit(0)

    if options.test:
        logger.config.level = 0
        failed = test(verbosity > 0)
        sys.exit(failed)

    if options.list:
        ids = sorted([key for key in _codecs if isinstance(key, int)])
        print "SHRINK codecs"
        print "----------------------------------------------------------------"
        print "id name         description"
        print "-- ------------ ------------------------------------------------"
        for id in ids:
            info = _codecs[id]
            layout = "{0:>2} {1:<12} {2:<48}" 
            print layout.format(info.id, info.name, info.doc or "")
        sys.exit(0)

    if not filenames:
        print help()
        sys.exit(1)
    if len(filenames) > 1:
        sys.exit("error: multiple filenames not supported")
    filename = filenames[0]

    codec = None
    codec_key = script.first(options.codec)
    if codec_key:
        try:
            codec_key = int(codec_key)
        except ValueError:
            pass
        try:
            codec = _codecs[codec_key]
        except KeyError:
            sys.exit("error: codec {0!r} not found".format(codec_key))
    
    parts = filename.split(".")
    if len(parts) == 1:
        raise ValueError("error: no filename extension found, use 'wav' or 'shk'.")
    else:
        basename = ".".join(parts[:-1])
        extension = parts[-1]

    if extension == "wav":
        if codec is None:
            id = max([id for id in _codecs if isinstance(id, int)])
            codec = _codecs[id]
        coder = codec.coder

        channels = wave.read(filename, scale=False)
        if len(shape(channels)) == 1:
            channels = channels.reshape((1, -1))
        stream = coder(channels)
        output = open(basename + ".shk", "w")
        output.write(stream.read(str))
        output.close()
    elif extension == "shk":          
        stream = BitStream(open(filename).read())
        header = stream.copy(6*8 + 8)
        if header.read(str, 6) != "SHRINK":
            logger.error("invalid format")
        else:
            logger.info("file with valid shrink format")
        id = header.read(uint8)
        logger.info("file encoded with shrink protocol {id}")
        if codec and codec.id != id:
             error = "file encoded with shrink coder {0}"
             raise ValueError(error.format(id))
        decoder = _codecs[id].decoder
        channels = decoder(stream)
        wave.write(channels, output=basename + ".wav")
    else:
        error = "unknown extension {0!r}, use 'wav' or 'shk'."
        raise ValueError(error.format(extension))
        
if __name__ == "__main__":
    main()

