"""A subclass of NesEnvironment for the Super Mario Bros ROM."""
import logging
import os
from gym import spaces
from .nes_env import NesEnv, MetaNesEnv


logger = logging.getLogger(__name__)


# (world_number, level_number, area_number, max_distance)
SMB_LEVELS = [
    (1, 1, 1, 3266), (1, 2, 2, 3266), (1, 3, 4, 2514), (1, 4, 5, 2430),
    (2, 1, 1, 3298), (2, 2, 2, 3266), (2, 3, 4, 3682), (2, 4, 5, 2430),
    (3, 1, 1, 3298), (3, 2, 2, 3442), (3, 3, 3, 2498), (3, 4, 4, 2430),
    (4, 1, 1, 3698), (4, 2, 2, 3266), (4, 3, 4, 2434), (4, 4, 5, 2942),
    (5, 1, 1, 3282), (5, 2, 2, 3298), (5, 3, 3, 2514), (5, 4, 4, 2429),
    (6, 1, 1, 3106), (6, 2, 2, 3554), (6, 3, 3, 2754), (6, 4, 4, 2429),
    (7, 1, 1, 2962), (7, 2, 2, 3266), (7, 3, 4, 3682), (7, 4, 5, 3453),
    (8, 1, 1, 6114), (8, 2, 2, 3554), (8, 3, 3, 3554), (8, 4, 4, 4989)
]


def encode_level(level: int) -> str:
    """
    Encode a level integer into the FCEUX opcode.

    Args:
        level: the level as a integer to decode

    Returns:
        a string of 3 integers:
        - world_number
        - level_number
        - area_number

    """
    world_number = int(level / 4) + 1
    level_number = (level % 4) + 1
    area_number = level_number
    # Worlds 1, 2, 4, 7 have a transition as area number 2 (so 2-2 is area 3 and 3, 2-3 is area 4, 2-4 is area 5)
    if world_number in [1, 2, 4, 7] and level_number >= 2:
        area_number += 1
    return '%d%d%d' % (world_number, level_number, area_number)


def is_int16(s: str) -> bool:
    """Return true if the input string is a 16-bit integer, false otherwise."""
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


class SuperMarioBrosEnv(NesEnv):
    """A Super Mario Bros level environment for Open.ai Gym."""

    def __init__(self,
        level: int=0,
        rom_path: str=None
    ) -> None:
        """
        Initialize a new SMB environment with the given level.

        Args:
            level: the number of the level to load
            rom_path: the path to the ROM for the game

        Returns:
            None

        """
        NesEnv.__init__(self)
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.level = level
        self.lua_path.append(os.path.join(package_directory, 'lua/super-mario-bros.lua'))
        self.launch_vars['target'] = encode_level(self.level)
        self.launch_vars['meta'] = '0'
        # check if a custom ROM path is employed
        if rom_path is not None:
            if not os.path.isfile(rom_path):
                raise ValueError('rom_path must be a valid Super Mario Bros. ROM')
        # fall-back on the default ROM
        else:
            rom_path = os.path.join(os.path.dirname(__file__), 'roms', 'super-mario.nes')
            if not os.path.isfile(rom_path):
                raise ValueError('default ROM ({}) missing.'.format(rom_path))
        self.rom_path = rom_path

    def _process_data_message(self, frame_number, data):
        # Format: data_<frame>#name_1:value_1|name_2:value_2|...
        if frame_number <= self.last_frame or self.info is None:
            return
        parts = data.split('|')
        for part in parts:
            if part.find(':') == -1:
                continue
            parts_2 = part.split(':')
            name = parts_2[0]
            value = int(parts_2[1])
            if 'is_finished' == name:
                self.is_finished = bool(value)
            elif 'distance' == name:
                self.reward = value - self.info[name]
                self.episode_reward = value
                self.info[name] = value
            else:
                self.info[name] = value

    def _process_screen_message(self, frame_number, data):
        # Format: screen_<frame>#<x (2 hex)><y (2 hex)><palette (2 hex)>|<x><y><p>|...
        if frame_number <= self.last_frame or self.screen is None:
            return
        parts = data.split('|')
        for part in parts:
            if 6 == len(part) and is_int16(part[0:2]) and is_int16(part[2:4]):
                x = int(part[0:2], 16)
                y = int(part[2:4], 16)
                self.screen[y][x] = self._get_rgb_from_palette(part[4:6])

    def _process_ready_message(self, frame_number):
        # Format: ready_<frame>
        if 0 == self.last_frame:
            self.last_frame = frame_number

    def _process_done_message(self, frame_number):
        # Done means frame is done processing, please send next command
        # Format: done_<frame>
        if frame_number > self.last_frame:
            self.last_frame = frame_number

    def _process_reset_message(self):
        # Reset means 'changelevel' needs to be sent and last_frame needs to be set to 0
        # Not implemented in non-meta levels
        pass

    def _process_exit_message(self):
        # Exit means fceux is terminating
        # Format: exit
        self.is_finished = True
        self._is_exiting = 1
        self.close()

    def _parse_frame_number(self, parts):
        # Parsing frame number
        try:
            frame_number = int(parts[1]) if len(parts) > 1 else 0
            return frame_number
        except:
            pass

        # Sometimes beginning of message is sent twice (screen_70screen_707#)
        if len(parts) > 2 and parts[2].isdigit():
            tentative_frame = int(parts[2])
            if self.last_frame - 10 < tentative_frame < self.last_frame + 10:
                return tentative_frame

        # Otherwise trying to make sense of digits
        else:
            digits = ''.join(c for c in ''.join(parts[1:]) if c.isdigit())
            tentative_frame = int(digits) if len(digits) > 1 else 0
            if self.last_frame - 10 < tentative_frame < self.last_frame + 10:
                return tentative_frame

        # Unable to parse - Likely an invalid message
        return None

    def _process_pipe_message(self, message):
        # Parsing
        parts = message.split('#')
        header = parts[0] if len(parts) > 0 else ''
        data = parts[1] if len(parts) > 1 else ''
        parts = header.split('_')
        message_type = parts[0] if len(parts) > 0 else ''
        frame_number = self._parse_frame_number(parts)

        # Invalid message - Ignoring
        if frame_number is None:
            return

        # Processing
        if 'data' == message_type:
            self._process_data_message(frame_number, data)
        elif 'screen' == message_type:
            self._process_screen_message(frame_number, data)
        elif 'ready' == message_type:
            self._process_ready_message(frame_number)
        elif 'done' == message_type:
            self._process_done_message(frame_number)
        elif 'reset' == message_type:
            self._process_reset_message()
        elif 'exit' == message_type:
            self._process_exit_message()

    def _get_reward(self):
        return self.reward

    def _get_episode_reward(self):
        return self.episode_reward

    def _get_is_finished(self):
        return self.is_finished

    def _get_state(self):
        return self.screen.copy()

    def _get_info(self):
        return self.info

    def _reset_info_vars(self):
        self.info = {
            'level': self.level,
            'distance': 0,
            'score': -1,
            'coins': -1,
            'time': -1,
            'player_status': -1
        }


class MetaSuperMarioBrosEnv(MetaNesEnv, SuperMarioBrosEnv):
    """A Super Mario Bros entire game environment for Open.ai Gym."""

    def __init__(self,
        average_over: int=10,
        passing_grade: int=600,
        min_tries_for_avg: int=5
    ) -> None:
        # initialize the MetaNes mixin
        MetaNesEnv.__init__(self,
            average_over=average_over,
            passing_grade=passing_grade,
            min_tries_for_avg=min_tries_for_avg,
            num_levels=32
        )
        # initialize the SMB mixin
        SuperMarioBrosEnv.__init__(self, level=0)
        # update the launch vars with the meta flag
        self.launch_vars['meta'] = '1'

    def _process_reset_message(self):
        self.last_frame = 0

    def _get_standard_reward(self, episode_reward):
        # Returns a standardized reward for an episode (i.e. between 0 and 1,000)
        min_score = 0
        target_score = float(SMB_LEVELS[self.level][-1]) - 40
        max_score = min_score + (target_score - min_score) / 0.99  # Target is 99th percentile (Scale 0-1000)
        std_reward = round(1000 * (episode_reward - min_score) / (max_score - min_score), 4)
        std_reward = min(1000, std_reward)  # Cannot be more than 1,000
        std_reward = max(0, std_reward)  # Cannot be less than 0
        return std_reward
