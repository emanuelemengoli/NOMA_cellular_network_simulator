import subprocess
subprocess.run(['python3', 'simulation_files/libs/install.py'])
import numpy as np
from numpy import argmax  
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statistics import median
from statistics import mean 
from geopy.distance import distance
from geopy.distance import geodesic
import math
import folium
from geopy.geocoders import Nominatim
import folium.plugins as plugins
from IPython.display import display
from matplotlib.animation import FuncAnimation,PillowWriter
import inspect
from functools import partial
from geopy.location import Location
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from numpy.random import uniform as U
import simpy
import logging
import random
import copy
import imageio
import os
from typing import Optional, List, Tuple
import sys
import geopandas as gpd
import datetime;
import warnings
from wdbo_criterion import *
from time import sleep, time
import gpytorch
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from wdbo_algo.optimizer import WDBOOptimizer
from pymobility.models.mobility import random_waypoint
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor,as_completed
import matplotlib as mpl
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import dill
import threading
import gc
import glob
from scipy.interpolate import interp1d

# Suppress warnings
warnings.filterwarnings("ignore")

