# ============================================================
# Adds a weather function call plugin to Whispering Tiger
#
# V0.0.2
#
# See https://github.com/Sharrnah/whispering
# ============================================================
#
import requests
import Plugins

class Phi4FunctionCallWeatherPlugin(Plugins.Base):

    tool = {
        "name": "get_weather",
        "description": "get weather",
        "parameters": {
            "city": {
                "description": "The name of the city",
                "type": "str",
                "default": "Berlin"
            }
        }
    }

    function_call_reply_prompt = 'You called {function_name} with the result temperature={temperature}, rain={rain}, showers={showers}, snowfall={snowfall}, precipitation={precipitation}, wind speed={wind_speed} to {function_description}.'

    def init(self):
        # prepare all possible settings
        self.init_plugin_settings(
            {
                "add_emoji": True,
            },
            settings_groups={
                "General": ["add_emoji"],
            }
        )
        pass

    def on_plugin_llm_function_registration_call(self, data_obj):
        if self.is_enabled(False):
            model = data_obj['model']
            task = data_obj['task']
            if model != "phi4":
                return None

            # set modified audio back to data_obj
            data_obj['tool_definition'] = self.tool
            return data_obj
        return None

    def on_plugin_llm_function_process_get_weather_call(self, data_obj):
        if self.is_enabled(False):
            model = data_obj['model']
            task = data_obj['task']
            if model != "phi4":
                return None

            raw_response = data_obj['response']
            function_name = data_obj['function_name']
            arguments = data_obj['arguments']

            ### do things here
            parameters = list(self.tool['parameters'].keys())
            result = self.return_weather_result(arguments[parameters[0]])

            # set modified object back to data_obj
            data_obj['type'] = 'llm_answer'

            # a reply_prompt is needed to answer the function call as LLM instead of just returning the result
            function_call_reply_prompt = self.function_call_reply_prompt.format(
                function_name=function_name,
                function_description=self.tool['description'],
                **result,
            )

            if self.get_plugin_setting("add_emoji"):
                function_call_reply_prompt += " Add an emoji for the weather like ⛅."

            data_obj['reply_prompt'] = function_call_reply_prompt

            return data_obj
        return None

    def return_weather_result(self, city_name):
        # get coordinates for city
        url = "https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=2&language=en&format=json".format(city_name=city_name)
        response = requests.get(url)
        response.raise_for_status()
        response_obj = response.json()
        latitude = response_obj['results'][0]['latitude']
        longitude = response_obj['results'][0]['longitude']

        # get weather for coordinates
        url = "https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,rain,showers,snowfall,precipitation,wind_speed_10m".format(latitude=latitude, longitude=longitude)
        response = requests.get(url)
        response.raise_for_status()
        response_obj = response.json()

        current_weather = response_obj['current']
        current_units = response_obj['current_units']

        result = {
            "temperature": str(current_weather['temperature_2m']) + " " + current_units['temperature_2m'],
            "rain": str(current_weather['rain']) + " " + current_units['rain'],
            "showers": str(current_weather['showers']) + " " + current_units['showers'],
            "snowfall": str(current_weather['snowfall']) + " " + current_units['snowfall'],
            "precipitation": str(current_weather['precipitation']) + " " + current_units['precipitation'],
            "wind_speed": str(current_weather['wind_speed_10m']) + " " + current_units['wind_speed_10m'],
        }

        return result
