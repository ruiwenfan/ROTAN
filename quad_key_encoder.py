import math
from nltk import ngrams

EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
Minlnggitude = -180
Maxlnggitude = 180

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlng2pxy(latitude, lnggitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    lnggitude = clip(lnggitude, Minlnggitude, Maxlnggitude)

    x = (lnggitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def latlng2quadkey(lat,lng,level):
    pixelX, pixelY = latlng2pxy(lat, lng, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY,level)
