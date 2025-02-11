import re
import os
import sys

from loguru import logger
from datetime import datetime
from xml.etree import ElementTree as ET

from .objects import Player, Game, Tournament
from .databases import GameDatabase, PlayerDatabase


def set_logging_level(level : str):
    """
    Sets the logging level for the logger.

    Args:
        - level (str): The logging level to set.
    """

    logger.remove()
    # log to console
    logger.add(sys.stdout, level=level)


def extract_tournament_trfx(folder : str) -> Tournament:
    """
    Extracts tournament information from a TRFX file.

    Args:
        - folder (str): The path to the folder containing the TRFX file.

    Returns:
        - Tournament: An instance of the Tournament class representing the extracted tournament information.
    """
    main_file_pattern = re.compile(r".*\.trfx")
    main_file = None
    for file in os.scandir(folder):
        if main_file_pattern.match(file.name):
            main_file = file
            break
    if main_file is None:
        logger.warn(f"No main file found in {folder}. You need to export the tournament file from Vega to get all metadata. Resorting to default")
        return Tournament(name='Unknown', date=datetime.now(), rounds=7, time_control='5+3')
    with open(main_file, 'r', encoding='latin1') as file:
        for i in range(0, 20):
            line = file.readline()
            if i == 0:
                name = line[4:].strip()
            if i == 3:
                date = line[4:].strip()
    date = datetime.strptime(date, '%d/%m/%Y')
    return Tournament(name, date, 7, '5+3')

def extract_tournament(folder : str) -> Tournament:
    """
    Extracts tournament information from a folder containing Vega XML files.

    Args:
        - folder (str): The path to the folder containing the Vega XML files.

    Returns:
        - Tournament: An instance of the Tournament class representing the extracted tournament information.
    """
    # in the folder, check for a file named *.vegx
    main_file_pattern = re.compile(r".*\.vegx")
    main_file = None
    for file in os.scandir(folder):
        if main_file_pattern.match(file.name):
            main_file = file
            break
    
    if main_file is None:
        logger.warning(f"No main file found in {folder}. You need to export the tournament file from Vega to get all metadata (file -> Export -> Tournament). Resorting to second best option.")
        tournament = extract_tournament_trfx(folder)
    else:
        # open the file and read the xml data
        with open(main_file, 'r') as file:
            data = file.read()
        root = ET.fromstring(data)

        # Extracting the tournament name
        tournament_name = root.find('Name').text if root.find('Name') is not None else 'Unknown'
        date_element = root.find('Date')
        begin_date = date_element.get('Begin') if date_element is not None else 'Unknown'
        rounds_number = int(root.find('.//RoundsNumber').text if root.find('.//RoundsNumber') is not None else 7)
        rate_move = root.find('.//RateMove').text if root.find('.//RateMove') is not None else 'Unknown'
        begin_date = datetime.strptime(begin_date, '%d/%m/%Y')
        tournament = Tournament(tournament_name, begin_date, rounds_number, rate_move)
    return tournament

def extract_players(folder : str) -> list[Player]:
    """
    Extracts players' information from a standings file in the specified folder.

    Args:
        - folder (str): The path to the folder containing the standings file.

    Returns:
        - list: A list of Player objects representing the extracted players.
    """
    file = os.path.join(folder, 'standings.qtf')
    players = []
    if not os.path.isfile(file):
        logger.debug(f"No standings file found in {folder}. Resorting to old naming convention.")
        file = os.path.join(folder, 'standing.qtf')
    with open(file, 'r') as file:
        data = file.read()
    lines = data.split('\n')
    tie_breaks = dict()
    tie_break_names = []
    doing_tie_break_names = False
    done_first_player = False
    current_player = None
    for i, line in enumerate(lines):
        if i == 14:
            name = line.replace(':: [s0;>*2', '').strip().replace(']', '').strip()
            tie_break_names.append(name)
            doing_tie_break_names = True
        elif doing_tie_break_names and not done_first_player:
            name = line.replace(':: [s0;>*2', '').replace(':: [s0;>2', '').strip().replace(']', '').strip()
            if name == '1' or name == '':
                doing_tie_break_names = False
            else:
                tie_break_names.append(name)
        if 's0;*' in line and '[s0;*2 NAME]' not in line:
            done_first_player = True
            name = line.split('s0;*2 ')[1][:-1]
            if '(' in name:
                name = name.split('(')[0].strip()
            player = Player(name)
            players.append(player)
            current_player = player.id
        if ':: [s0;>2' in line and done_first_player:
            value = float(line.split(':: [s0;>2')[1].split(']')[0].strip())
            if current_player in tie_breaks and len(tie_breaks[current_player]) < len(tie_break_names):
                tie_breaks[current_player][tie_break_names[len(tie_breaks[current_player])]] = value
            elif current_player not in tie_breaks:
                tie_breaks[current_player] = {tie_break_names[0]: value}
    return players, tie_breaks, tie_break_names


def extract_games(folder : str, tournament : Tournament, 
                  game_database : GameDatabase, player_database : PlayerDatabase, 
                  add_home_advantage : bool = True, forfeit_keep_points : bool = True):
    """
    Extracts games from pairings files in the specified folder and adds them to the tournament, game database, and player database.

    Args:
        - folder (str): The path to the folder containing the pairings files.
        - tournament (Tournament): The tournament object to which the extracted games will be added.
        - game_database (GameDatabase): The game database object to which the extracted games will be added.
        - player_database (PlayerDatabase): The player database object used to retrieve player information.
        - forfeit_keep_points (bool, optional): If True, points associated with each player are counted as the given points in case of a forfeit. This allows for custom match results that do not fit the normal points system.

    """
    for round in range(1, tournament.rounds + 1):
        file = os.path.join(folder, f'pairings{round}.qtf')
        if not os.path.isfile(file):
            logger.debug(f"No pairings file found for round {round} in {folder}. Resorting to old naming convention.")
            file = os.path.join(folder, f'pairs-bis{round}.qtf')
        if not os.path.isfile(file):
            tournament.rounds = round - 1
            break
        with open(file, 'r') as file:
            data = file.read()
        data = data.split('\n')[0]
        local = data.split('::')
        # re of only alpha characters and spaces
        for game in range(20, len(local) - 9, 10):
            if not '(not paired)' in local[game + 8]:
                white_player = local[game + 2]
                # get the player name by only keeping the alpha characters and spaces
                white_player = re.sub(r'[\[\]=@123456789]', '', white_player).strip()
                white_player = player_database.get_player_by_name(white_player)
                result = local[game + 5][7:-1].replace(' ', '')
                result = result.replace('\u00bd', '1/2')
                black_player = local[game + 8]
                black_player = re.sub(r'[\[\]=@123456789]', '', black_player).strip()

                if black_player == '( bye )':
                    logger.debug(f"Bye found for {white_player.name}")
                    tournament.add_bye(white_player.id, round)
                    continue

                black_player = player_database.get_player_by_name(black_player)
                if white_player is not None and black_player is not None:
                    game = Game(white_player.id, black_player.id, result, date=tournament.get_date(), 
                                tournament_id=tournament.id, add_home_advantage=add_home_advantage, 
                                forfeit_keep_points=forfeit_keep_points)
                    game_database.add(game)
