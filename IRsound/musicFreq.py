import math

from color import Color

'''
primitives for music to frequency manipulation
'''


class MusicFreq:
    baseC0 = 261.62556530061  # C0
    log2BaseC0 = math.log2(baseC0)
    LOG_baseC0 = math.log(baseC0)  # 5.5669143414923
    MUSICAL_INC = 2 ** (1 / 12)  # 1.0594630943593 // 2 ^ (1 / 12)
    LOG_MUSICAL_INC = math.log(MUSICAL_INC)  # 0.0577622650466
    LOG2 = math.log(2)  # 0.6931471805599

    musicalInc = 2. ** (1. / 12.)  # 1.0594630943593;  # 2 ** (1 / 12)
    note_str = {'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'}

    def noteoct2freq(note: int, oct: int) -> float:
        return MusicFreq.baseC0 * (MusicFreq.musicalInc ** (note + 12. * oct))

    def freq2octave(f: float, oct: int = 0) -> float:  # 2^(log2(f)-log2base)
        return f / (2. ** (int(math.log2(f)) - MusicFreq.log2BaseC0))

    def not2string(note: int) -> str:
        return MusicFreq.note_str[note % 12]

    def freq2note(freq: float) -> float:
        if freq <= 0: return 0
        return int((math.log(freq) - MusicFreq.LOG_baseC0) / MusicFreq.LOG_MUSICAL_INC -
                   MusicFreq.freq2oct(freq) * 12.)

    def freq2oct(freq: float) -> int:
        if freq <= 0: return -999
        return int(math.log2(freq) - MusicFreq.log2BaseC0)

    def freq2color(freq: float) -> int:
        f0 = MusicFreq.baseC0
        fz = MusicFreq.baseC0 * 2
        freq = MusicFreq.freq2octave(freq, 0)
        ratio = (freq - f0) / (fz - f0)
        return Color.interpolate(0xff0000, 0x0011ff, ratio)

    def freq2ColorCCS(freq: float) -> str:  # '#rrggbb hex format
        return Color.CCS(MusicFreq.freq2color(freq))
