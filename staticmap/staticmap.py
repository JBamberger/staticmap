import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, CancelledError
from io import BytesIO
from math import sqrt, log, tan, pi, cos, ceil, floor, atan, sinh
from typing import Tuple, Callable, Union

from PIL import Image, ImageDraw
from PIL.Image import Resampling
from requests import RequestException
from requests_cache import CachedSession


class AntialiasShape(ABC):
    def draw(self, canvas: ImageDraw, coord_to_pix: Callable[[float, float], Tuple[float, float]]):
        raise NotImplementedError()


class DirectShape(ABC):
    def draw(self, canvas: Image, coord_to_pix: Callable[[float, float], Tuple[float, float]]):
        raise NotImplementedError()


class Line(AntialiasShape):
    def __init__(self, coords, color, width, simplify=True):
        """
        Line that can be drawn in a static map

        :param coords: an iterable of lon-lat pairs, e.g. ((0.0, 0.0), (175.0, 0.0), (175.0, -85.1))
        :type coords: list
        :param color: color suitable for PIL / Pillow
        :type color: str
        :param width: width in pixel
        :type width: int
        :param simplify: whether to simplify coordinates, looks less shaky, default is true
        :type simplify: bool
        """
        self.coords = coords
        self.color = color
        self.width = width
        self.simplify = simplify

    @property
    def extent(self):
        """
        calculate the coordinates of the envelope / bounding box: (min_lon, min_lat, max_lon, max_lat)

        :rtype: tuple
        """
        return (
            min((c[0] for c in self.coords)),
            min((c[1] for c in self.coords)),
            max((c[0] for c in self.coords)),
            max((c[1] for c in self.coords)),
        )

    def draw(self, canvas: ImageDraw, coord_to_pix: Callable[[float, float], Tuple[float, float]]):
        points = [coord_to_pix(*coords) for coords in self.coords]

        if self.simplify:
            points = _simplify(points)

        for point in points:
            # draw extra points to make the connection between lines look nice
            canvas.ellipse((
                point[0] - self.width + 1, point[1] - self.width + 1,
                point[0] + self.width - 1, point[1] + self.width - 1
            ), fill=self.color)

        canvas.line(points, fill=self.color, width=self.width * 2)


class CircleMarker(AntialiasShape):
    def __init__(self, coord, color, width):
        """
        :param coord: a lon-lat pair, eg (175.0, 0.0)
        :type coord: tuple
        :param color: color suitable for PIL / Pillow
        :type color: str
        :param width: marker width
        :type width: int
        """
        self.coord = coord
        self.color = color
        self.width = width

    @property
    def extent_px(self):
        return (self.width,) * 4

    def draw(self, canvas: ImageDraw, coord_to_pix: Callable[[float, float], Tuple[float, float]]):
        point = coord_to_pix(*self.coord)
        canvas.ellipse((
            point[0] - self.width, point[1] - self.width,
            point[0] + self.width, point[1] + self.width
        ), fill=self.color)


class IconMarker(DirectShape):
    def __init__(self, coord, file_path, offset_x, offset_y):
        """
        :param coord:  a lon-lat pair, eg (175.0, 0.0)
        :type coord: tuple
        :param file_path: path to icon
        :type file_path: str
        :param offset_x: x position of the tip of the icon. relative to left bottom, in pixel
        :type offset_x: int
        :param offset_y: y position of the tip of the icon. relative to left bottom, in pixel
        :type offset_y: int
        """
        self.coord = coord
        self.img = Image.open(file_path, 'r')
        self.offset = (offset_x, offset_y)

    @property
    def extent_px(self):
        w, h = self.img.size
        return (
            self.offset[0],
            h - self.offset[1],
            w - self.offset[0],
            self.offset[1],
        )

    def draw(self, canvas: Image, coord_to_pix: Callable[[float, float], Tuple[float, float]]):
        x, y = coord_to_pix(*self.coord)
        position = (x - self.offset[0], y - self.offset[1])

        canvas.paste(self.img, position, self.img)


class Polygon(AntialiasShape):
    """
    Polygon that can be drawn on map

    :param coords: an iterable of lon-lat pairs, e.g. ((0.0, 0.0), (175.0, 0.0), (175.0, -85.1))
    :type coords: list
    :param fill_color: color suitable for PIL / Pillow, can be None (transparent)
    :type fill_color: str
    :param outline_color: color suitable for PIL / Pillow, can be None (transparent)
    :type outline_color: str
    :param simplify: whether to simplify coordinates, looks less shaky, default is true
    :type simplify: bool
    """

    def __init__(self, coords, fill_color, outline_color, simplify=True):
        self.coords = coords
        self.fill_color = fill_color
        self.outline_color = outline_color
        self.simplify = simplify

    @property
    def extent(self):
        return (
            min((c[0] for c in self.coords)),
            min((c[1] for c in self.coords)),
            max((c[0] for c in self.coords)),
            max((c[1] for c in self.coords)),
        )

    def draw(self, canvas: ImageDraw, coord_to_pix: Callable[[float, float], Tuple[float, float]]):
        points = [coord_to_pix(*coord) for coord in self.coords]

        if self.simplify:
            points = _simplify(points)

        if self.fill_color or self.outline_color:
            canvas.polygon(points, fill=self.fill_color, outline=self.outline_color)


def _lon_to_x(lon, zoom):
    """
    transform longitude to tile number
    :type lon: float
    :type zoom: int
    :rtype: float
    """
    if not (-180 <= lon <= 180):
        lon = (lon + 180) % 360 - 180

    return ((lon + 180.) / 360) * pow(2, zoom)


def _lat_to_y(lat, zoom):
    """
    transform latitude to tile number
    :type lat: float
    :type zoom: int
    :rtype: float
    """
    if not (-90 <= lat <= 90):
        lat = (lat + 90) % 180 - 90

    return (1 - log(tan(lat * pi / 180) + 1 / cos(lat * pi / 180)) / pi) / 2 * pow(2, zoom)


def _y_to_lat(y, zoom):
    return atan(sinh(pi * (1 - 2 * y / pow(2, zoom)))) / pi * 180


def _x_to_lon(x, zoom):
    return x / pow(2, zoom) * 360.0 - 180.0


def _simplify(points, tolerance=11):
    """
    :param points: list of lon-lat pairs
    :type points: list
    :param tolerance: tolerance in pixel
    :type tolerance: float
    :return: list of lon-lat pairs
    :rtype: list
    """
    if not points:
        return points

    new_coords = [points[0]]

    for p in points[1:-1]:
        last = new_coords[-1]

        dist = sqrt(pow(last[0] - p[0], 2) + pow(last[1] - p[1], 2))
        if dist > tolerance:
            new_coords.append(p)

    new_coords.append(points[-1])

    return new_coords


class StaticMap:
    def __init__(self,
                 width,
                 height,
                 padding_x=0,
                 padding_y=0,
                 url_template="http://a.tile.komoot.de/komoot-2/{z}/{x}/{y}.png",
                 tile_size=256,
                 tile_request_timeout=None,
                 headers=None,
                 reverse_y=False,
                 background_color="#fff",
                 delay_between_retries=0,
                 concurrent_connections=2,
                 max_retries=3):
        """
        :param width: map width in pixel
        :type width: int
        :param height:  map height in pixel
        :type height: int
        :param padding_x: min distance in pixel from map features to border of map
        :type padding_x: int
        :param padding_y: min distance in pixel from map features to border of map
        :type padding_y: int
        :param url_template: tile URL
        :type url_template: str
        :param tile_size: the size of the map tiles in pixel
        :type tile_size: int
        :param tile_request_timeout: time in seconds to wait for requesting map tiles
        :type tile_request_timeout: float
        :param headers: additional headers to add to http requests
        :type headers: dict
        :param reverse_y: tile source has TMS y origin
        :type reverse_y: bool
        :param background_color: Image background color, only visible when tiles are transparent
        :type background_color: str
        :param delay_between_retries: number of seconds to wait between retries of map tile requests
        :type delay_between_retries: int
        :param concurrent_connections: Number of concurrent connections to use for tile downloads.
        :type concurrent_connections: int
        :param max_retries: Max numbers of retries per tile
        :type max_retries: int
        """
        self.width = width
        self.height = height
        self.padding = (padding_x, padding_y)
        self.url_template = url_template
        self.headers = headers
        self.tile_size = tile_size
        self.request_timeout = tile_request_timeout
        self.reverse_y = reverse_y
        self.background_color = background_color

        # features
        self.shapes = []

        # fields that get set when map is rendered
        self.x_center = 0
        self.y_center = 0
        self.zoom = 0

        self.concurrent_connections = concurrent_connections
        self.delay_between_retries = delay_between_retries
        self.max_retries = max_retries

    def add_shape(self, shape: Union[AntialiasShape, DirectShape]):
        self.shapes.append(shape)

    def add_line(self, line: Line):
        self.add_shape(line)

    def add_marker(self, marker: Union[IconMarker, CircleMarker]):
        self.add_shape(marker)

    def add_polygon(self, polygon: Polygon):
        self.add_shape(polygon)

    def render(self, zoom=None, center=None):
        """
        render static map with all map features that were added to map before

        :param zoom: optional zoom level, will be optimized automatically if not given.
        :type zoom: int
        :param center: optional center of map, will be set automatically from markers if not given.
        :type center: list
        :return: PIL image instance
        :rtype: Image.Image
        """

        if not self.shapes and not (center and zoom):
            raise RuntimeError("cannot render empty map, add lines / markers / polygons first")

        if zoom is None:
            self.zoom = self._calculate_zoom()
        else:
            self.zoom = zoom

        if center:
            self.x_center = _lon_to_x(center[0], self.zoom)
            self.y_center = _lat_to_y(center[1], self.zoom)
        else:
            # get extent of all lines
            extent = self.determine_extent(zoom=self.zoom)

            # calculate center point of map
            lon_center, lat_center = (extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2
            self.x_center = _lon_to_x(lon_center, self.zoom)
            self.y_center = _lat_to_y(lat_center, self.zoom)

        image = Image.new('RGB', (self.width, self.height), self.background_color)

        self._draw_base_layer(image)
        self._draw_features(image)

        return image

    def determine_extent(self, zoom=None):
        """
        calculate common extent of all current map features

        :param zoom: optional parameter, when set extent of markers can be considered
        :type zoom: int
        :return: extent (min_lon, min_lat, max_lon, max_lat)
        :rtype: tuple
        """
        extents = []
        for shape in self.shapes:
            try:
                extents.append(shape.extent)
            except AttributeError:
                e = (shape.coord[0], shape.coord[1])

                if zoom is None:
                    extents.append(e * 2)
                    continue

                # consider dimension of marker
                e_px = shape.extent_px

                x = _lon_to_x(e[0], zoom)
                y = _lat_to_y(e[1], zoom)

                extents += [(
                    _x_to_lon(x - float(e_px[0]) / self.tile_size, zoom),
                    _y_to_lat(y + float(e_px[1]) / self.tile_size, zoom),
                    _x_to_lon(x + float(e_px[2]) / self.tile_size, zoom),
                    _y_to_lat(y - float(e_px[3]) / self.tile_size, zoom)
                )]

        return (
            min(e[0] for e in extents),
            min(e[1] for e in extents),
            max(e[2] for e in extents),
            max(e[3] for e in extents)
        )

    def _calculate_zoom(self):
        """
        calculate the best zoom level for given extent

        :param extent: extent in lon lat to render
        :type extent: tuple
        :return: lowest zoom level for which the entire extent fits in
        :rtype: int
        """

        for z in range(17, -1, -1):
            extent = self.determine_extent(zoom=z)

            width = (_lon_to_x(extent[2], z) - _lon_to_x(extent[0], z)) * self.tile_size
            if width > (self.width - self.padding[0] * 2):
                continue

            height = (_lat_to_y(extent[1], z) - _lat_to_y(extent[3], z)) * self.tile_size
            if height > (self.height - self.padding[1] * 2):
                continue

            # we found first zoom that can display entire extent
            return z

        # map dimension is too small to fit all features
        return 0

    def geo_to_tile(self, coords):
        return _lon_to_x(coords[0], self.zoom), _lat_to_y(coords[1], self.zoom)

    def tile_to_img(self, coords):
        px = (coords[0] - self.x_center) * self.tile_size + self.width / 2
        py = (coords[1] - self.y_center) * self.tile_size + self.height / 2
        return int(round(px)), int(round(py))

    def _get_tile_url(self, x, y) -> str:
        # x and y may have crossed the date line
        max_tile = 2 ** self.zoom
        tile_x = (x + max_tile) % max_tile
        tile_y = (y + max_tile) % max_tile

        if self.reverse_y:
            tile_y = ((1 << self.zoom) - tile_y) - 1

        return self.url_template.format(z=self.zoom, x=tile_x, y=tile_y)

    def _draw_base_layer(self, image: Image):
        x_min = int(floor(self.x_center - (0.5 * self.width / self.tile_size)))
        y_min = int(floor(self.y_center - (0.5 * self.height / self.tile_size)))
        x_max = int(ceil(self.x_center + (0.5 * self.width / self.tile_size)))
        y_max = int(ceil(self.y_center + (0.5 * self.height / self.tile_size)))

        # assemble all map tiles needed for the map
        tiles = [(x, y, self._get_tile_url(x, y))
                 for x in range(x_min, x_max) for y in range(y_min, y_max)]

        with CachedSession(cache_name='tile_cache', cache_control=True) as session:
            def download_tile(url):
                res = session.get(url, timeout=self.request_timeout, headers=self.headers)
                return res.status_code, res.content

            thread_pool = ThreadPoolExecutor(self.concurrent_connections)

            for nb_retry in range(self.max_retries):
                if not tiles:
                    break
                if nb_retry > 0 and self.delay_between_retries:
                    # to avoid stressing the map tile server too much, wait some seconds
                    time.sleep(self.delay_between_retries)

                failed_tiles = []
                futures = [thread_pool.submit(download_tile, tile[2]) for tile in tiles]
                for tile, future in zip(tiles, futures):
                    x, y, url = tile

                    try:
                        status_code, response_content = future.result()
                    except CancelledError:
                        print('Cancelled download externally.')
                        break
                    except RequestException:
                        status_code, response_content = None, None

                    if status_code != 200:
                        print("request failed [{}]: {}".format(status_code, url))
                        failed_tiles.append(tile)
                        continue

                    tile_image = Image.open(BytesIO(response_content)).convert("RGBA")
                    box = [*self.tile_to_img((x, y)),
                           *self.tile_to_img((x + 1, y + 1))]
                    image.paste(tile_image, box, tile_image)

                # put failed back into list of tiles to fetch in next try
                tiles = failed_tiles

        if tiles:
            raise RuntimeError("could not download {} tiles: {}".format(len(tiles), tiles))

    def _draw_features(self, canvas: Image.Image):

        # Pillow does not support anti aliasing for lines and circles
        # There is a trick to draw them on an image that is twice the size and resize it at the end
        # before it gets merged with  the base layer

        def coord_to_px(lon, lat) -> Tuple[float, float]:
            return self.tile_to_img(self.geo_to_tile((lon, lat)))

        def aa_coord_to_px(lon, lat) -> Tuple[float, float]:
            x, y = coord_to_px(lon, lat)
            return x * 2, y * 2

        aa_image = Image.new('RGBA', (self.width * 2, self.height * 2), (255, 0, 0, 0))
        aa_canvas = ImageDraw.Draw(aa_image)

        for shape in filter(lambda s: isinstance(s, AntialiasShape), self.shapes):
            shape.draw(aa_canvas, aa_coord_to_px)

        aa_image = aa_image.resize((self.width, self.height), Resampling.LANCZOS)
        # merge lines with base image
        canvas.paste(aa_image, (0, 0), aa_image)

        for shape in filter(lambda s: isinstance(s, DirectShape), self.shapes):
            shape.draw(canvas, coord_to_px)


if __name__ == '__main__':
    map = StaticMap(300, 400, 10)
    line = Line([(13.4, 52.5), (2.3, 48.9)], 'blue', 3)
    map.add_line(line)
    image = map.render()
    image.save('berlin_paris.png')
