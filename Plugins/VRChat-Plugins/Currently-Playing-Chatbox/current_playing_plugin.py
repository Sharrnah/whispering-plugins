# ============================================================
# Shows currently playing Song over OSC using Whispering Tiger
# Version 1.0.5
# See https://github.com/Sharrnah/whispering
# ============================================================
import datetime

import Plugins

import VRC_OSCLib
import settings
import asyncio

from winsdk.windows.media.control import GlobalSystemMediaTransportControlsSessionManager

PROMPT = {
    "command": ["playing", "listening", "listens", "song", "music", "track", "current"]
}

PLAYERS = ["Spotify", "iTunes", "AmazonMusic", "Deezer", "YouTubeMusic", "WindowsMediaPlayer", "Groove", "Music", "Zune", "Winamp", "AIMP", "foobar2000", "Spotube"]


class CurrentPlayingPlugin(Plugins.Base):
    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "timer": True,
                "commands": None,
                "players": None,
                "only_playing": False,
                # progression UI settings
                "display_progressbar": True,
                "display_time_progress": True,
                "display_progress_percentage": True,
                "progress_bar_length": 11,
                "progress_bar_pos_character": "🐅",
            },
            settings_groups={
                "General": ["timer", "commands", "players", "only_playing"],
                "Progression": ["display_progressbar", "display_time_progress", "display_progress_percentage", "progress_bar_length", "progress_bar_pos_character"],
            }
        )

    def is_allowed_player(self, source_app_user_model_id):
        # try to get player names from settings
        player_names = self.get_plugin_setting("players")
        if player_names is None:
            player_names = PLAYERS

        # check if player name is in source_app_user_model_id
        for player_name in player_names:
            if player_name.lower() in source_app_user_model_id.lower():
                return True
        return False

    def create_progress_bar(self, percentage, length=10):
        progress_bar_pos_character = self.get_plugin_setting("progress_bar_pos_character", "🐅")

        block_percentage = 100 / length  # Calculate the average percentage of each block
        filled_blocks = int((percentage / block_percentage))  # Calculate the number of filled blocks based on the block percentage
        if filled_blocks >= length - 1 and percentage >= ((length - 1) * block_percentage + block_percentage / 2):  # If filled_blocks is equal to or greater than length - 1 and percentage is halfway through the last block
            filled_blocks = length - 1  # Set filled_blocks to length - 1 to ensure the circle is within the progress bar
        empty_blocks = length - filled_blocks - 1  # Subtract 1 to account for the circle
        #progress_bar = '┣' + '━' * filled_blocks + '●' + '━' * empty_blocks + '┫'
        progress_bar = '┣' + '━' * filled_blocks + progress_bar_pos_character + '━' * empty_blocks + '┫'
        return progress_bar

    def format_time(self, time):
        total_seconds = int(time.total_seconds())
        time_obj = datetime.timedelta(seconds=total_seconds)

        if total_seconds >= 3600:  # If total_duration is an hour or more
            hours = str(time_obj.seconds // 3600)
            minutes = str((time_obj.seconds % 3600) // 60).zfill(2)
            seconds = str((time_obj.seconds % 3600) % 60).zfill(2)
            time_str = f"{hours}:{minutes}:{seconds}"
        else:  # If total_duration is less than an hour
            minutes = str(time_obj.seconds // 60).zfill(2)
            seconds = str(time_obj.seconds % 60).zfill(2)
            time_str = f"{minutes}:{seconds}"

        return time_str

    async def get_current_song(self):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")
        only_playing = self.get_plugin_setting("only_playing")
        display_progressbar = self.get_plugin_setting("display_progressbar")
        display_time_progress = self.get_plugin_setting("display_time_progress")
        display_progress_percentage = self.get_plugin_setting("display_progress_percentage")
        progress_bar_length = self.get_plugin_setting("progress_bar_length")

        if self.is_enabled(False):
            manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
            sessions = manager.get_sessions()
            for session in sessions:
                info = await session.try_get_media_properties_async()
                #print(info.title, info.artist, session.source_app_user_model_id)
                if info and self.is_allowed_player(session.source_app_user_model_id):
                    status = session.get_playback_info()

                    progress_bar_string = ""
                    progress_time_string = ""
                    # get playback position
                    timeline_properties = session.get_timeline_properties()
                    playback_position = timeline_properties.position
                    total_duration = timeline_properties.end_time - timeline_properties.start_time
                    if total_duration > datetime.timedelta(0):
                        progress_percentage = (playback_position / total_duration) * 100
                    else:
                        progress_percentage = 0
                    if display_time_progress:
                        playback_position_str = self.format_time(playback_position)
                        total_duration_str = self.format_time(total_duration)
                        progress_time_string = f"\n{playback_position_str} / {total_duration_str}"
                    if display_progressbar:
                        progress_bar = self.create_progress_bar(progress_percentage, length=progress_bar_length)
                        progress_bar_string = f"\n{progress_bar}"
                        if display_progress_percentage:
                            progress_bar_string += f"{int(progress_percentage)}%"
                    elif display_progress_percentage:
                        progress_bar_string = f"\n{int(progress_percentage)}%"

                    if only_playing and status.playback_status.name == 'PLAYING' or not only_playing:
                        VRC_OSCLib.Chat(f"{status.playback_status.name}🎵: {info.title} by {info.artist}{progress_bar_string}{progress_time_string}",
                                        True, False,
                                        osc_address, IP=osc_ip, PORT=osc_port,
                                        convert_ascii=False)
                    return info.title + " by " + info.artist
        return None

    def timer(self):
        if self.get_plugin_setting("timer", True):
            asyncio.run(self.get_current_song())
        pass

    def stt(self, text, result_obj):
        plugin_commands = self.get_plugin_setting("commands")
        if plugin_commands is None:
            plugin_commands = PROMPT['command']

        question = text.strip().lower()

        # return with current playing song if command word is found
        if any(ele in question for ele in plugin_commands):
            asyncio.run(self.get_current_song())

        return

    def tts(self, text, device_index, websocket_connection=None, download=False):
        return
