# Snake AI project

## Overview

The project involves using AI to play the game of Snake, with version 3 being the stable release. Other versions have not yet had their parameters tuned, rendering the AI ineffective in gameplay.
## Configuration

Default parameter values are specified in `setting.py`, while custom parameter values can be defined in `par_lev.json`. Note that default scores for rewards are set in `game.py`.

## Usage

1. **Play with Pre-trained Model:**
   ```bash
   python version3_Linear_QNet/test_main.py

This command will use the pre-trained model(trained for 1000 games) for gameplay.
   
2. **Retrain the Model:**
   ```bash
   python version3_Linear_QNet/main.py

Running this command will initiate the retraining process, overwriting any existing models.but be aware that this action will overwrite any previously trained models.

