#!/usr/bin/env python

def test_shrink_v0():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v0(data)
"""

def test_grow_v0():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v0(data)
    >>> audio.shrink.grow_v0(stream)
"""

def test_shrink_v1():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v1(data)
"""

def test_grow_v1():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v1(data)
    >>> audio.shrink.grow_v1(stream)
"""

def test_shrink_v2():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v2(data)
"""

def test_grow_v2():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v2(data)
    >>> audio.shrink.grow_v2(stream)
"""

def test_shrink_v3():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v3(data)
"""

def test_grow_v3():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v3(data)
    >>> audio.shrink.grow_v3(stream)
"""

def test_shrink_v4():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v4(data)
"""

def test_grow_v4():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v4(data)
    >>> audio.shrink.grow_v4(stream)
"""

def test_shrink_v5():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v5(data)
"""

def test_grow_v5():
    """
    >>> import audio.shrink
    >>> import audio.wave
    >>> import numpy as np
    >>> dt = 1.0 / 44100.0
    >>> f = 440.0
    >>> T = 1.0
    >>> t = np.r_[0.0:T:dt]
    >>> data = np.cos(2.0 * np.pi * f * t)
    >>> data = audio.wave.read(audio.wave.write(data), scale=False)
    >>> stream = audio.shrink.shrink_v5(data)
    >>> audio.shrink.grow_v5(stream)
"""


if __name__ == "__main__":
    import docbench
    docbench.benchmod(profile=True)



