import os
import pickle

import pandas as pd
import numpy as np


def clean_inputs(inputs: dict) -> dict:
    return inputs


def populate(league: str, conditions: str, teams: list, output_type: str):
    """
    Generate a json friendly dataframe of sports-bettors outputs
    """
    assert league in ['college_football', 'nfl']

