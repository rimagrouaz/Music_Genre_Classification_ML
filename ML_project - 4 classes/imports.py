import IPython.display as ipd
from matplotlib import pyplot as plt
import librosa
import librosa.display
import os
import time
import numpy as np
import scipy.io.wavfile
import scipy.fft
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import joblib
import yt_dlp
import subprocess
import requests
import re
from pydub import AudioSegment
from pydub.effects import normalize

