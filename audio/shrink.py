#!/usr/bin/env python
# coding: utf-8

"""
SHRINK Lossless Audio coder
"""

# Python 2.7 Standard Library
from __future__ import division
import argparse
import inspect
import sys

# Third-Party Libraries
import numpy as np
import numpy.linalg as la

# Digital Audio Coding Library
import audio.bitstream
import audio.coders
import audio.frames
import audio.wave
import breakpoint
import logfile

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
from .about_shrink import *

#
# Coders Registration
# ------------------------------------------------------------------------------
#
class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

_coders = {}

def register(id, name, coder, decoder, doc=None):
    """
    Register a SHRINK coder pair by id (number) and name.
    """
    info = struct(id=id, name=name, coder=coder, decoder=decoder, doc=doc)
    _coders[id] = _coders[name] = info

#
# Breakpoint Handler and Decorator
# ------------------------------------------------------------------------------
#
def log_remaining():
    def handler(**kwargs):
        remaining = kwargs.get("remaining")
        logfile.tag("audio.shrink")
        logfile.info("time remaining: {remaining:.1f} secs.")
    return handler

log_ETA = breakpoint.function(on_yield=log_remaining, progress=True, dt=10.0)

#
# BitStream Helper
# ------------------------------------------------------------------------------
#
def byte_pad(stream):
    extra = len(stream) % 8
    if extra:
        stream.write((8 - extra) * [False])

#
# Version 0 - Amplitude Rice Coding
# ------------------------------------------------------------------------------
#
@log_ETA
def shrink_v0(channels):
    channels = np.array(channels, ndmin=2)
    stream = audio.bitstream.BitStream()
    stream.write("SHRINK")
    version = 0
    stream.write(version, np.uint8)
    length = np.shape(channels)[1]
    stream.write(length, np.uint32)
    stereo = (np.shape(channels)[0] == 2)
    stream.write(stereo)
    for i_channel, channel in enumerate(channels):
        if np.size(channel):
            rice_tag = audio.coders.rice.from_frame(channel, signed=True)
            stream.write(rice_tag.b, np.uint8)
            i = 0
            count, stop = 0, 1
            while i < length:
                i_next = min(i + 4410, length)
                stream.write(channel[i:i_next], rice_tag)
                i = i_next
                if count >= stop:
                    count = 0
                    progress = (i_channel + i / length) / len(channels)
                    x = yield (progress, stream)
                    if x is not None:
                        stop = stop * x
                count += 1
    byte_pad(stream)
    yield (1.0, stream)

@log_ETA
def grow_v0(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(np.uint8) == 0
    length = stream.read(np.uint32)
    num_channels = stream.read(bool) + 1
    channels = np.zeros((num_channels, length), dtype=np.int16)
    if np.size(channels):        
        for i_channel in range(num_channels):
            b = stream.read(np.uint8)
            rice_tag = audio.coders.rice(b=b, signed=True)
            i = 0
            count, stop = 0, 1
            while i < length:
                i_next = min(i + 4410, length)
                channels[i_channel][i:i_next] = stream.read(rice_tag, i_next - i)
                i = i_next
                if count >= stop:
                    count = 0
                    progress = (i_channel + i / length) / len(channels)  
                    x = yield (progress, channels)
                    if x is not None:
                        stop = stop * x
                count += 1
    assert all(np.r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(0, "v0", shrink_v0, grow_v0, "amplitude rice coder")

#
# Version 1 - Differential Rice Coding
# ------------------------------------------------------------------------------
#
@log_ETA
def shrink_v1(channels):
    channels = np.array(channels, ndmin=2)
    stream = audio.bitstream.BitStream()
    stream.write("SHRINK")
    version = 1
    stream.write(version, np.uint8)
    length = np.shape(channels)[1]
    stream.write(length, np.uint32)
    stereo = (np.shape(channels)[0] == 2)
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if np.size(channel):
            channel = np.array(channel, dtype=np.int32)
            delta = np.diff(np.r_[0, channel])
            rice_tag = audio.coders.rice.from_frame(delta, signed=True)
            stream.write(rice_tag.b, np.uint8)
            i = 0
            while i < length:
                if count >= stop:
                    count = 0
                    progress = (i_channel + i / length) / len(channels)
                    x = (yield progress, stream)
                    if x is not None:
                        stop = stop * x
                count += 1
                i_next = min(i + 4410, length)
                stream.write(delta[i:i_next], rice_tag)
                i = i_next
    byte_pad(stream)
    yield (1.0, stream)

@log_ETA
def grow_v1(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(np.uint8) == 1
    length = stream.read(np.uint32)
    num_channels = stream.read(bool) + 1
    channels = np.zeros((num_channels, length), dtype=np.int16)
    count, stop = 0, 1
    for i in range(num_channels):
        if np.size(channels[i]):
            b = stream.read(np.uint8)
            rice_tag = audio.coders.rice(b=b, signed=True)
            delta = np.zeros(length, dtype=np.int32)
            j = 0
            while j < length:
                if count >= stop:
                    count = 0
                    progress = (i + j / length) / num_channels
                    x = (yield progress, channels)
                    if x is not None:
                        stop = stop * x
                count += 1
                j_next = min(j + 4410, length)  
                delta[j:j_next] = stream.read(rice_tag, j_next - j)
                j = j_next
            channels[i] = np.cumsum(delta)
    assert all(np.r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(1, "v1", shrink_v1, grow_v1, "differential rice coder")

#
# Version 2 - First-Order Prediction Residual Rice Coder
# ------------------------------------------------------------------------------
#
@log_ETA
def shrink_v2(channels):
    channels = np.array(channels, ndmin=2)
    stream = audio.bitstream.BitStream()
    stream.write("SHRINK")
    version = 2
    stream.write(version, np.uint8)
    length = np.shape(channels)[1]
    stream.write(length, np.uint32)
    stereo = (np.shape(channels)[0] == 2)
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if np.size(channel):
            channel = np.array(channel, dtype=np.int32)
            delta = np.diff(np.r_[0, np.diff(np.r_[0, channel])])
            rice_tag = audio.coders.rice.from_frame(delta, signed=True)
            stream.write(rice_tag.b, np.uint8)
            i = 0
            while i < length:
                if count >= stop:
                    count = 0
                    progress = (i_channel + i / length) / len(channels)
                    x = (yield progress, stream)
                    if x is not None:
                        stop = stop * x
                count += 1
                i_next = min(i + 4410, length) 
                stream.write(delta[i:i_next], rice_tag)
                i = i_next
    byte_pad(stream)
    yield (1.0, stream)

@log_ETA
def grow_v2(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(np.uint8) == 2
    length = stream.read(np.uint32)
    num_channels = stream.read(bool) + 1
    channels = np.zeros((num_channels, length), dtype=np.int16)
    count, stop = 0, 1
    for i in range(num_channels):
        if np.size(channels[i]):
            n = stream.read(np.uint8)
            rice_tag = audio.coders.rice(n, signed=True)
            delta = np.zeros(length, dtype=np.int32)
            j = 0
            while j < length:
                if count >= stop:
                    count = 0
                    progress = (i + j / length) / num_channels
                    x = (yield progress, channels)
                    if x is not None:
                        stop = stop * x
                count += 1
                j_next = min(j + 4410, length) 
                delta[j:j_next] = stream.read(rice_tag, j_next - j)
                j = j_next
            channels[i] = np.cumsum(np.cumsum(delta))
    assert all(np.r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(2, "v2", shrink_v2, grow_v2, "1st-order pred. residual rice coder")

#
# Version 3 - Polynomial Prediction Residual Rice Coder
# ------------------------------------------------------------------------------
#
@log_ETA
def shrink_v3(channels, N=14):
    channels = np.array(channels, ndmin=2)
    stream = audio.bitstream.BitStream()
    stream.write("SHRINK")
    version = 3
    stream.write(version, np.uint8)
    length = np.shape(channels)[1]
    stream.write(length, np.uint32)
    num_channels = np.shape(channels)[0]
    stereo = (num_channels == 2)
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if np.size(channels):
            channel = np.array(channel, dtype=np.int32)
            rice_tag = audio.coders.rice.from_frame(channel, signed=True)
            i = 0
            while i <= N:
                delta = np.diff(np.r_[0, channel])
                new_rice_tag = audio.coders.rice.from_frame(delta, signed=True)
                if new_rice_tag.b >= rice_tag.b:
                    break
                else:
                    rice_tag = new_rice_tag
                    channel = delta
                    i += 1
            stream.write(i, audio.coders.rice(3, signed=False))
            stream.write(rice_tag.b, np.uint8)
            j = 0
            while j < length:
                if count >= stop:
                    count = 0
                    progress = (i_channel + j / length) / num_channels  
                    x = (yield progress, stream)
                    if x is not None:
                        stop = x * stop
                count += 1
                j_next = min(j + 4410, length)        
                stream.write(channel[j:j_next], rice_tag)
                j = j_next
    byte_pad(stream)
    yield (1.0, stream)

@log_ETA
def grow_v3(stream):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(np.uint8) == 3
    length = stream.read(np.uint32)
    num_channels = stream.read(bool) + 1
    channels = np.zeros((num_channels, length), dtype=np.int16)
    count, stop = 0, 1
    for j in range(num_channels):
        if np.size(channels[j]):
            i = stream.read(audio.coders.rice(3, signed=False))
            n = stream.read(np.uint8)
            rice_tag = audio.coders.rice(n, signed=True)
            delta = np.zeros(length, dtype=np.int32)
            k = 0
            while k < length:
                if count >= stop:
                    count = 0
                    progress = (j + k / length) / num_channels
                    x = (yield progress, channels)
                    if x is not None:
                        stop = stop * x
                count += 1
                k_next = min(k + 4410, length)
                delta[k:k_next] = stream.read(rice_tag, k_next - k)
                k = k_next
            for _ in range(i):
                delta = np.cumsum(delta)
            channels[j] = delta
    assert all(np.r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels)

register(3, "v3", shrink_v3, grow_v3, "polynomial pred. residual rice coder")

#
# Version 4 - Polynomial Prediction Residual Rice Coder Within Frames
# ------------------------------------------------------------------------------
#
@log_ETA
def shrink_v4(channels, N=14, frame_length=882):
    channels = np.array(channels, ndmin=2)
    stream = audio.bitstream.BitStream()
    stream.write("SHRINK")
    version = 4
    stream.write(version, np.uint8)
    length = np.shape(channels)[1]
    stream.write(length, np.uint32)
    stereo = (np.shape(channels)[0] == 2)
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if np.size(channel):
            channel = np.array(channel, dtype=np.int32)
            frames = audio.frames.split(channel, frame_length, pad=True)
            tail = length % frame_length
            for i_frame, frame in enumerate(frames):
                if count >= stop:
                    count = 0
                    progress = (i_channel + i_frame / len(frames)) / len(channels)
                    x = (yield progress, stream)
                    if x is not None:
                        stop = stop * x                
                count += 1
                rice_tag = audio.coders.rice.from_frame(frame, signed=True)
                i = 0
                while i <= N:
                    delta = np.diff(np.r_[0, frame])
                    new_rice_tag = audio.coders.rice.from_frame(delta, signed=True)
                    if new_rice_tag.b >= rice_tag.b:
                        break
                    else:
                        rice_tag = new_rice_tag
                        frame = delta
                    i += 1
                stream.write(i, audio.coders.rice(3, signed=False))
                stream.write(rice_tag.b, np.uint8)
                stream.write(frame, rice_tag)
    byte_pad(stream)
    yield (1.0, stream)

@log_ETA
def grow_v4(stream, N=14, frame_length=882):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(np.uint8) == 4
    length = stream.read(np.uint32)
    num_frames = (length // frame_length) + (length % frame_length != 0)
    pad = num_frames * frame_length - length
    num_channels = stream.read(bool) + 1
    channels = np.zeros((num_channels, length + pad), dtype=np.int16)
    count, stop = 0, 1
    for j in range(num_channels):
        channel = channels[j]
        if np.size(channel):
            channel = np.array(channel, dtype=np.int32)
            for i_frame in range(num_frames):
                i_array = i_frame * frame_length
                if count >= stop:
                    count = 0
                    progress = (j + i_frame / num_frames) / num_channels
                    x = (yield progress, channels[:,:length])
                    if x is not None:
                        stop = stop * x
                count += 1
                i = stream.read(audio.coders.rice(3, signed=False))
                n = stream.read(np.uint8)
                delta = stream.read(audio.coders.rice(n, signed=True), frame_length)
                #print i, n, delta
                delta = np.array(delta, dtype=np.int32)
                for _ in range(i):
                    delta = np.cumsum(delta)
                channels[j][i_array:i_array+frame_length] = delta 
    assert all(np.r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels[:,:length])

doc ="polynomial pred. residual rice coder within frames"
register(4, "v4", shrink_v4, grow_v4, doc)

#
# Version 5 - Polynomial Prediction Residual Rice Coder Within Overlapping Frames
# ------------------------------------------------------------------------------
#
@log_ETA
def shrink_v5(channels, N=14, frame_length=882):
    channels = np.array(channels, ndmin=2)
    stream = audio.bitstream.BitStream()
    stream.write("SHRINK")
    version = 5
    stream.write(version, np.uint8)
    length = np.shape(channels)[1]
    stream.write(length, np.uint32)
    stereo = (np.shape(channels)[0] == 2)
    stream.write(stereo)
    count, stop = 0, 1
    for i_channel, channel in enumerate(channels):
        if np.size(channel) > 0:
            channel = np.array(channel, dtype=np.int32)
            frames = audio.frames.split(\
                       np.r_[np.zeros(N + 1, dtype=np.int32), channel], 
                       frame_length = frame_length + N + 1, 
                       overlap = N + 1,
                       pad=True)
            for i_frame, frame in enumerate(frames):
                if count >= stop:
                    count = 0
                    progress = (i_channel + i_frame / len(frames)) / len(channels)
                    x = (yield progress, stream)
                    if x is not None:
                        stop = stop * x
                count += 1
                rice_tag = audio.coders.rice.from_frame(frame, signed=True)
                i = 0
                while i <= N:
                    delta = np.diff(frame)
                    new_rice_tag = audio.coders.rice.from_frame(delta[N-i:], signed=True)
                    if new_rice_tag.b >= rice_tag.b:
                        break
                    else:
                        rice_tag = new_rice_tag
                        frame = delta
                    i += 1
                stream.write(i, audio.coders.rice(3, signed=False))
                stream.write(rice_tag.b, np.uint8)
                stream.write(frame[N+1-i:], rice_tag)
    byte_pad(stream)
    yield (1.0, stream)

@log_ETA
def grow_v5(stream, N=14, frame_length=882):
    assert stream.read(str, 6) == "SHRINK"
    assert stream.read(np.uint8) == 5
    length = stream.read(np.uint32)
    num_channels = stream.read(bool) + 1
    num_frames = (length // frame_length) + (length % frame_length != 0)
    pad = num_frames * frame_length - length
    channels = np.zeros((num_channels, length + pad), dtype=np.int16)
    count, stop = 0, 1
    for j in range(num_channels):
        if np.size(channels):
            channel = np.zeros((0,), dtype=np.int16)
            for i_frame in range(num_frames):
                i_array = i_frame * frame_length
                if count >= stop:
                    count = 0
                    progress = (j + i_frame / num_frames) / num_channels
                    x = (yield progress, channels[:,:length])
                    if x is not None:
                        stop = stop * x
                count += 1
                i = stream.read(audio.coders.rice(3, signed=False))
                n = stream.read(np.uint8)
                delta = stream.read(audio.coders.rice(n, signed=True), frame_length)
                delta = np.array(delta, dtype=np.int32)
                sum_offset = []
                if i > 0:
                    if np.size(channel):
                        start = channel[-i:].astype(np.int32)
                    else:
                        start = np.zeros((i,), dtype=np.int32)
                    for k in range(i):
                       sum_offset.insert(0, start[-1])
                       start = np.diff(start)
                    for k in range(i):
                        delta = np.cumsum(delta) + sum_offset[k]
                channel = np.r_[channel, delta]

            channels[j] = channel[:length+pad]
    assert all(np.r_[stream.read(bool, len(stream))] == False)
    yield (1.0, channels[:,:length])

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
          version of the rice_tags.

    >>> dataset = [ 
    ...             [], 
    ...             [[]],
    ...             [[], []], 
    ...             [0], 
    ...             [[0], [0]], 
    ...             [0, 0], 
    ...             [[0, 0], [0, 0]],
    ...             [-2**15, -2**15+1, -2, -1, 0, 1, 2**15 -2, 2**15 - 1], 
    ...             [[-2**15, 0, 2**15-1], [0, 1, 2]],
    ...           ]
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> t = np.r_[0.0:1.0:dt]
    >>> f = 440.0
    >>> data = np.round(np.cos(2*np.pi*f*t)).astype(np.int16)
    >>> dataset.append(data)
    >>> dataset.append([data, data])

    >>> def check_round_trip(shrink, grow):
    ...     checks =  []
    ...     for data in dataset:
    ...         checks.append((data == grow(shrink(data))).all())
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

#
# Command-Line Interface
# ------------------------------------------------------------------------------
#

def main():
    "Command-Line Entry Point"

    description = "SHRINK Lossless Audio Coder"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("filename", nargs="?", type=str,
                         help = "filename (WAVE or SHRINK file")
    parser.add_argument("-v", "--verbose", 
                        action  = "count", 
                        default = 0,
                        help    = "display more information")
    parser.add_argument("-s", "--silent",
                        action  = "count", 
                        default = 0,
                        help    = "display less information")
    parser.add_argument("-c", "--coder",
                        type = str,
                        help = "select a rice_tag (by id or name)")
    parser.add_argument("-l", "--list", action="store_true",
                        help = "list available coders")
    args = parser.parse_args()

    verbosity = args.verbose - args.silent
    logfile.config.level = verbosity
    def format(logfile, message, tag, date):
        tag = tag or ""
        return " {0:<9} | {1:<18} | {2}\n".format(logfile.name, tag, message)
    logfile.config.format = format

    if args.list:
        ids = sorted([key for key in _coders if isinstance(key, int)])
        print "SHRINK Coders"
        print "----------------------------------------------------------------"
        print "id name         description"
        print "-- ------------ ------------------------------------------------"
        for id in ids:
            info = _coders[id]
            layout = "{0:>2} {1:<12} {2:<48}" 
            print layout.format(info.id, info.name, info.doc or "")
        sys.exit(0)

    rice_tag = args.coder
    if coder:
        try:
            coder = int(coder)
        except ValueError:
            pass
        try:
            coder = _coders[coder]
        except KeyError:
            sys.exit("error: coder {0!r} not found".format(coder_key))

    filename = args.filename
    if filename is None:
        return
    parts = filename.split(".")
    if len(parts) == 1:
        raise ValueError("error: no filename extension found, use 'wav' or 'shk'.")
    else:
        basename = ".".join(parts[:-1])
        extension = parts[-1]

    if extension == "wav":
        if coder is None:
            id = max([id for id in _coders if isinstance(id, int)])
            coder = _coders[id]
        coder = coder.coder

        channels = audio.wave.read(filename, scale=False)
        if len(np.shape(channels)) == 1:
            channels.reshape((1, -1))
        stream = coder(channels)
        output = open(basename + ".shk", "w")
        output.write(stream.read(str))
        output.close()
    elif extension == "shk":          
        stream = audio.bitstream.BitStream(open(filename).read())
        header = stream.copy(6*8 + 8)
        if header.read(str, 6) != "SHRINK":
            logfile.error("invalid format")
        else:
            logfile.info("file with valid shrink format")
        id = header.read(np.uint8)
        logfile.info("file encoded with shrink protocol {id}")
        if coder and coder.id != id:
             error = "file encoded with shrink coder {0}"
             raise ValueError(error.format(id))
        decoder = _coders[id].decoder
        channels = decoder(stream)
        audio.wave.write(channels, output=basename + ".wav")
    else:
        error = "unknown extension {0!r}, use 'wav' or 'shk'."
        raise ValueError(error.format(extension))


