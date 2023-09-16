import random
import sys

from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
from way_calculation import way

class test1():
    def __init__(self):
        self.a = 100
        self.b =3
    def com(self, a, b):
        if a >b:
            print("a>b")
        else:
            print("a<b")