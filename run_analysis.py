"""Reproduce every dataset, table, chart, and memo in the required order."""

from data_exploration import run_exploration
from statistical_analysis import run_statistical_analysis


if __name__ == "__main__":
    run_exploration()
    run_statistical_analysis()
