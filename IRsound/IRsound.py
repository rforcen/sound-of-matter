import zipfile
from glob import iglob as idir
from math import pi, sin
from os.path import join as joinPath

import jcamp
import matplotlib.pyplot as plt
import numpy  as np
from scipy.io.wavfile import write as wav_write

from musicFreq import MusicFreq
from peaks import peakdetect


def jcamp2wav(fh, wavenme=None, rate=44100, secs=5):
    ' convert a fh to a .dx file to a .wav file'

    def sound_func(amp, hz, t):  # sound generation
        return amp * sin(hz * t)

    def mean(l):
        return sum(l) / len(l)

    dct = jcamp.jcamp_read(fh)

    x, y = dct['x'], dct['y']
    dif = (y.max() - y.min())

    pk = peakdetect(y, x, lookahead=1, delta=dif / 10)

    max_peaks = pk[0]  # [ [x0,y0] , ...., [xn,yn] ]

    waves = [[(_y - y.min()) / dif, MusicFreq.freq2octave(_x, 0)] for _x, _y in max_peaks]  # amp(0..1), freq oct '0'
    waves.sort(reverse=True)  # get <= 10 most powerful
    waves = waves[:10]
    pi2 = pi * 2  # -> evaluate waves average for each sample
    data = np.asarray(
        [mean([sound_func(amp, hz, t) for amp, hz in waves]) for t in np.arange(0, secs * pi2, pi2 / rate)],
        dtype=np.float32)

    if wavenme is None or not wavenme:
        wavenme = fh.name.replace('.dx', '.wav')

    wav_write(wavenme, rate, data)


def plot_jcamp(base_path='jcamp', fnme='Acenapthene.dx'):
    dct = jcamp.JCAMP_reader(joinPath(base_path, fnme))

    # plot
    fig = plt.figure(fnme)
    plt.plot(dct['x'], dct['y'])
    plt.show()


def plot_jcamp_dir(base_path='jcamp'):
    def sizeInches(wPix, hPix, dpi=96):  # from pixels to inches in 96 dpi
        return (wPix / dpi, hPix / dpi)  # 96 dpi

    fig = plt.figure('jcamp-dx ' + base_path, sizeInches(1500, 800))
    fig.subplots_adjust(left=0.03, bottom=.05, wspace=0.12, hspace=0.45, right=0.97, top=0.96)
    plt.rcParams.update({'font.size': 5})

    dct = []
    for nf, fnme in enumerate(idir(joinPath(base_path, '*dx'), recursive=True)):
        dct = jcamp.JCAMP_reader(fnme)
        if nf < 25:
            plt.title(dct['title'][:30])
            plt.subplot(5, 5, nf + 1)
            plt.plot(dct['x'], dct['y'])
    plt.show()


def testunicodes():  # to solve unicode conversion issues in org jcamp.py
    '''
    line 39: # add 'b' binary mode in open
        with open(filename, 'rb') as filehandle:
    line 69: decode to utf-8 ignoring conversion errors
        line=line.decode('utf-8','ignore')
    '''
    with open('/Users/asd/Documents/_voicesync/spectrum/jcamp/Emodine.dx', 'rb') as f:
        for line in f:
            line = line.decode('utf-8', 'ignore').strip()
            print(line)


def j2w_test():
    with open('Maleicanhydride.dx', 'rb') as fh:
        jcamp2wav(fh, wavenme=None, rate=44100, secs=5)


def jcamzip2wav(dxfn, znme='jcamp.zip'):
    'read .dx files from .zip generating .wav in current folder'
    with zipfile.ZipFile(znme) as z:
        zdxfn = 'jcamp/' + dxfn
        if zdxfn in z.namelist():
            jcamp2wav(z.open(zdxfn), dxfn.replace('.dx', '.wav'))


if __name__ == '__main__':
    jcamzip2wav('Maleicanhydride.dx', 'jcamp.zip')
