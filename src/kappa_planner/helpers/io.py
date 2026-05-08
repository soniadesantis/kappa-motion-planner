"""
Input/output utilities for corridor data.

This module provides functions to load corridor configurations from
external file formats such as YAML and XML.
"""

import yaml
from math import pi

from ..corridor import CorridorWorld
from .corridor_functions import get_corridor_from_vector


def read_corridors_from_xml(file_name):
    '''
    Read the corridors from an xml file.

    :param file_name: name of the xml file
    :type file_name: string

    :return: list of corridors
    :rtype: list of CorridorWorld
    '''
    from xml.dom import minidom
    corridor_list = []
    #  each time the string 'inkscape:label="corridor' appears, print the next charaters until the next double quote
    corridor_number = 0
    with open(file_name) as f:  
        for line in f:
            if 'inkscape:label="corridor' in line:
                corridor_number = int(line[line.find('inkscape:label="corridor')+len('inkscape:label="corridor'):line.find('"', line.find('inkscape:label="corridor')+len('inkscape:label="corridor')+1)])
                corridor_width = float(width) + float(stroke_width)
                corridor_height = float(height) + float(stroke_width)
                corridor_x = float(x) - float(stroke_width)/2 + corridor_width * 0.5
                corridor_y = (-1) * (float(y) - float(stroke_width)/2 + corridor_height * 0.5)
                # print(f'Corridor {corridor_number} has width {corridor_width}, height {corridor_height}, x {corridor_x}, y {corridor_y}, stroke_width {stroke_width}')
                corridor_list.append(CorridorWorld(width = corridor_width, height = corridor_height, center = [corridor_x, corridor_y], tilt = pi*0.5, label = f'corridor{corridor_number}', number = int(corridor_number) ))
                
                # if corridor_width >= corridor_height:
                #     corridor_list.append(CorridorWorld(width = corridor_height, height = corridor_width, center = [corridor_x, corridor_y], tilt = pi/2 ))
                # else:
                #     corridor_list.append(CorridorWorld(width = corridor_width, height = corridor_height, center = [corridor_x, corridor_y], tilt = 0 ))

            if 'stroke-width:' in line:
                stroke_width = line[line.find('stroke-width:')+len('stroke-width:'):line.find(';', line.find('stroke-width:')+len('stroke-width:')+1)]
            if 'width=' in line:
                width = line[line.find('width="')+len('width="'):line.find('"', line.find('width="')+len('width="')+1)]
            if 'height=' in line:
                height = line[line.find('height="')+len('height="'):line.find('"', line.find('height="')+len('height="')+1)]
            if 'x=' in line:
                x = line[line.find('x="')+len('x="'):line.find('"', line.find('x="')+len('x="')+1)]
            if 'y=' in line:
                y = line[line.find('y="')+len('y="'):line.find('"', line.find('y="')+len('y="')+1)]
    corridor_list_ordered = sorted(corridor_list, key = lambda corridor: corridor.number)
    return corridor_list_ordered


def read_corridors_from_yaml(file_path):
    '''
    Read the corridors from a yaml file.
    
    :param file_path: path to the yaml file
    :type file_path: string
    
    :return: list of corridors
    :rtype: list of CorridorWorld
    '''
    corridor_list = []
    with open(file_path, 'r') as file:
        corridor_library = yaml.safe_load(file)
        for corridor in corridor_library['Corridors']:
            corridor_list.append(get_corridor_from_vector([x / 1000 for x in corridor['tail']], [x / 1000 for x in corridor['head']], corridor['width']/1000, add_height = 0))

            # corridor_list.append(get_corridor_from_vector(corridor['tail']/1000, corridor['head']/1000, corridor['width']/1000, add_height = 0))
    return corridor_list


def read_corridors_from_yaml_scale(file_path, scale = 1):
    '''
    Read the corridors from a yaml file.
    
    :param file_path: path to the yaml file
    :type file_path: string

    :param scale: scale factor
    :type scale: float
    
    :return: list of corridors
    :rtype: list of CorridorWorld
    '''
    corridor_list = []
    with open(file_path, 'r') as file:
        corridor_library = yaml.safe_load(file)
        for corridor in corridor_library['Corridors']:
            corridor_list.append(get_corridor_from_vector([x * scale for x in corridor['tail']], [x * scale for x in corridor['head']], corridor['width'] * scale, add_height = 0))

            # corridor_list.append(get_corridor_from_vector(corridor['tail']/1000, corridor['head']/1000, corridor['width']/1000, add_height = 0))
    return corridor_list


def read_corridors_from_yaml_cwht(file_path, scale = 1):
    '''
    Read the corridors from a yaml file.
    
    :param file_path: path to the yaml file
    :type file_path: string

    :param scale: scale factor
    :type scale: float
    
    :return: list of corridors
    :rtype: list of CorridorWorld
    '''
    corridor_list = []
    with open(file_path, 'r') as file:
        corridor_library = yaml.safe_load(file)
        for corridor in corridor_library['Corridors']:
            corridor_list.append(CorridorWorld(width = corridor['width'] * scale,\
                                               height = corridor['height'] * scale, \
                                               center = [x * scale for x in corridor['center']], \
                                               tilt = corridor['tilt'],\
                                               label = corridor['id']))

    return corridor_list


def svg_to_corridor_dict(corridor_map):
    # load an SVG file
    from xml.dom import minidom
    # read from a given svg file and print the first 1000 characters
    print(minidom.parse('corridor_map.svg').toprettyxml()[:1000])