import colorsys
import struct

'''
color basic manipulation
'''


class Color:
    def interpolateHLS(org=0x0000ff, dest=0xff00ff, ratio=0.5):  # return list r,g,b
        def color2arbgFloat(col: int) -> list:  # list of a,r,g,b
            return [f / 255. for f in reversed(list(struct.pack('=i', int(col))))]

        a, r, g, b = color2arbgFloat(org)
        hslOrg = colorsys.rgb_to_hls(r, g, b)
        a, r, g, b = color2arbgFloat(dest)
        hslDest = colorsys.rgb_to_hls(r, g, b)
        # H1+(H2-H1)*Ratio, L1+(L2-L1)*Ratio, S1+(S2-S1)*Ratio
        rgb = [c[0] + (c[1] - c[0]) * ratio for c in zip(hslOrg, hslDest)]
        return rgb


    def color2arbg(col: int) -> list:  # list of a,r,g,b
        return list(struct.pack('=i', int(col)))

    def color2arbgFloat(col: int) -> list:  # list of a,r,g,b in 0..1 float values
        l = list(struct.pack('=i', int(col)))
        return list(reversed([c / 255. for c in l]))

    def arbg2color(a: int, r: int, g: int, b: int) -> int:
        return Color.arbgflist2color([a, r, g, b])

    def arbgflist2color(l: list) -> int:  # float list to arbg
        return int.from_bytes(struct.pack('=4B', *list(map(int, l))), byteorder='little')

    def interpolate(org=0x0000ff, dest=0xff0000,
                    ratio=0) -> int:  # org, dest in RRGGBB ratio 0..1 -> return list[r,g,b]
        return Color.arbgflist2color(
            [c[0] + (c[1] - c[0]) * ratio + 0.5 for c in zip(Color.color2arbg(org), Color.color2arbg(dest))])

    def interpolateList(ratio=0, org=0x0000ff,
                        dest=0xff0000, ) -> list:  # org, dest in RRGGBB ratio 0..1 -> return list[r,g,b]
        c = Color.arbgflist2color(
            [c[0] + (c[1] - c[0]) * ratio + 0.5 for c in zip(Color.color2arbg(org), Color.color2arbg(dest))])
        return Color.color2arbgFloat(c)

    def colIndex(col: int, ix: int) -> float: return struct.pack('=i', int(col))[ix % 4] / 255.

    def alphaF(col: int) -> float: return Color.colIndex(col, 0)

    def redF(col: int) -> float: return Color.colIndex(col, 1)

    def greenF(col: int) -> float: return Color.colIndex(col, 2)

    def blueF(col: int) -> float: return Color.colIndex(col, 3)

    def brightness(col: int) -> float:
        return sum([f * Color.colIndex(col, ix) for ix, f in enumerate([0.299, 0.587, 0.114])])

    def CCS(col: int) -> str:
        return "#{0:06x}".format(col & 0xffffff)
