import os.path
import time

from staticmap import *


def show_pos():
    m = StaticMap(200, 200)

    m.add_shape(CircleMarker((10, 47), 'white', 18))
    m.add_shape(CircleMarker((10, 47), '#0036FF', 12))

    m.render(zoom=5).save('marker.png')


def show_line():
    m = StaticMap(200, 200, 80)

    coordinates = [[12.422, 45.427], [13.749, 44.885]]

    m.add_shape(Line(coordinates, 'white', 6))
    m.add_shape(Line(coordinates, '#D2322D', 4))

    m.render().save('ferry.png')


def show_icon_marker():
    from pathlib import Path
    sample_dir = Path(__file__).parent.parent / 'samples'

    m = StaticMap(240, 240, 80)

    m.add_shape(IconMarker((6.63204, 45.85378), str(sample_dir / 'icon-flag.png'), 12, 32))
    m.add_shape(IconMarker((6.6015, 45.8485), str(sample_dir / 'icon-factory.png'), 18, 18))

    m.render().save('icons.png')


def draw_polygon():
    m = StaticMap(1024, 1024, padding_x=80)

    m.add_shape(Polygon(
        [
            [12.422, 45.427],
            [13.749, 45.427],
            [13.749, 44.885],
            [12.422, 44.885],
        ],
        outline_color='#00ff00',
        fill_color='#00ff00',
        simplify=True,
    ))
    m.render().save('polygon.png')


def draw_uncached():
    m = StaticMap(1024, 1024, padding_x=80, cache_file=None)

    m.add_shape(Polygon(
        [
            [12.422, 45.427],
            [13.749, 45.427],
            [13.749, 44.885],
            [12.422, 44.885],
        ],
        outline_color='#00ff00',
        fill_color='#00ff00',
        simplify=True,
    ))
    m.render().save('polygon.png')


def test_caching():
    cache_file = 'tmp_cache'
    cache_file_name = cache_file + '.sqlite'

    if os.path.exists(cache_file_name):
        os.remove(cache_file_name)

    for i in range(10):
        start = time.time()
        m = StaticMap(1024, 1024, padding_x=80, cache_file=cache_file)

        m.add_shape(Polygon(
            [
                [12.422, 45.427],
                [13.749, 45.427],
                [13.749, 44.885],
                [12.422, 44.885],
            ],
            outline_color='#00ff00',
            fill_color='#00ff00',
            simplify=True,
        ))
        m.render()

        print(time.time() - start)

    os.remove(cache_file_name)


if __name__ == '__main__':
    show_pos()
    show_line()
    show_icon_marker()
    draw_polygon()
    draw_uncached()

    test_caching()
