import os
import sys
import argparse
import requests
import queue
import pyaudio
import wave
import threading
import time
import shutil
import json
import keyboard

from lxml import html
from melo.api import TTS
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class TTSMelo:
    def __init__(self, language='ZH', device = 'auto', speed=1.0):
        self.language = language
        self.device = device
        self.speed = speed
        self.model = TTS(language=self.language, device=device)
        self.speaker_ids = self.model.hps.data.spk2id

    def generate_audio(self, text, output_path):
        self.model.tts_to_file(
            text, 
            self.speaker_ids[self.language ], 
            output_path, 
            speed=self.speed)
        

class TextReader(threading.Thread):
    def __init__(self, text_file, play_queue, tts, remove_cache=True):
        super().__init__()
        self.stop_flag = False
        self.text_file = text_file
        self.play_queue = play_queue
        self.tts = tts
        self.pause_flag = False

        self.tmp_folder = 'tmp_wav_text_reader'
        if remove_cache:
            if os.path.exists(self.tmp_folder):
                shutil.rmtree(self.tmp_folder)
            os.mkdir(self.tmp_folder)
    
    def run(self):
        def __replace_chars_with_newline(s, chars="！？。"):
            for char in chars:
                s = s.replace(char, f'{char}\n')
            return s
        
        with open(self.text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"No text found in the text file {self.text_file}")
            
            index_l = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                chunked_lines = __replace_chars_with_newline(line).split('\n')
                index_cl = 0
                for c_line in chunked_lines:
                    if self.stop_flag:
                        break
                    if c_line == '' or c_line.isspace():
                        continue
                    output_path = os.path.join(self.tmp_folder, f'{os.path.splitext(os.path.basename(self.text_file))[0]}-{index_l}-{index_cl}.wav')
                    try:
                        self.tts.generate_audio(c_line, output_path)
                    except Exception as e:
                        print(f"Error occurred while generating audio for line: {c_line}")
                        print(e)
                        continue
                    self.play_queue.put((output_path, c_line))
                    while self.pause_flag:
                        time.sleep(0.5)
                        if self.stop_flag:
                            break
                    index_cl += 1
                index_l += 1

        print("Text reader stopped")
    
    def stop(self):
        self.stop_flag = True

    def pause(self):
        self.pause_flag = not self.pause_flag
                

class WebReader(threading.Thread):
    def __init__(self, web_url, play_queue, tts, xpath_config):
        super().__init__()
        self.web_url = web_url
        self.play_queue = play_queue
        self.tts = tts
        self.stop_flag = False
        self.pause_flag = False
        self.xpath_config = xpath_config

        urlparse_result = urlparse(self.web_url)   
        self.web_url = f"{urlparse_result.scheme}://{urlparse_result.netloc}"
        self.href = web_url.replace(self.web_url, '')
        self.xpath_config = xpath_config.get(urlparse_result.hostname, None)
        if not self.xpath_config:
            raise ValueError(f"Xpath configuration not found for the url: {self.web_url}")
        
        self.tmp_folder = 'tmp_html_text_reader'
        if os.path.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)
        os.mkdir(self.tmp_folder)


        tmp_folder = 'tmp_wav_text_reader'
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder)

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=chrome_options)

        self.text_reader = None


    def run(self):
        def __get_title(html):
            eles = content.xpath(self.xpath_config['title'])
            if eles:
                return eles[0].text_content()
            return None
        
        def __get_article(html):
            eles = content.xpath(self.xpath_config['article'])
            if eles:
                return ''.join([i.text_content() for i in eles])
            return None
        
        def __get_next_page(html):
            eles = content.xpath(self.xpath_config['next_page'])
            if eles:
                return eles[0].attrib['href']
            return None

        while(self.href and not self.stop_flag):
            try:
                print(f"Fetching the web page: {self.web_url}/{self.href}")
                res = requests.get(f"{self.web_url}{self.href}")
                res.raise_for_status()
            except Exception as e:
                print(f"Error occurred while fetching the web page: {self.web_url}")
                print(e)
                break

            content = res.content
            content = html.fromstring(content)
            artical = __get_article(content)
            if not artical:
                print(f"No article found for the web page: {self.web_url}/{self.href}")
                print(f"Trying to fetch the web page using selenium")
                self.driver.get(f"{self.web_url}{self.href}")
                html_c = self.driver.page_source
                content = html.fromstring(html_c)
                artical = __get_article(content)

            if artical:
                title = __get_title(content)
                if not title:
                    title = f"article_{time.time()}"
                with open(os.path.join(self.tmp_folder, f'{title}.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{title}{artical}")

                print(f"Reading the article: {title}")
                self.text_reader = TextReader(os.path.join(self.tmp_folder, f'{title}.txt'), self.play_queue, self.tts, False)
                self.text_reader.start()

                while self.text_reader.is_alive():
                    if self.stop_flag:
                        self.text_reader.stop()
                        self.text_reader.join(0.5)
                        break

                    time.sleep(0.5)
            else:
                print(f"No article found for the web page: {self.web_url}/{self.href}")
            
            while not self.stop_flag and not self.play_queue.qsize() <= 5:
                time.sleep(0.5)
            self.href = __get_next_page(content)

        print("Web reader stopped")

    def stop(self):
        self.stop_flag = True
        try:
            self.driver.quit()
        except:
            pass

    def pause(self):
        self.pause_flag = not self.pause_flag
        if self.text_reader:
            self.text_reader.pause()
            

class AudioPlayer(threading.Thread):
    def __init__(self, play_queue, callback=None):
        super().__init__()
        self.play_queue = play_queue
        self.stop_flag = False
        self.p_audio = None
        self.stream = None
        self.pause_flag = False
        self.callback = callback

    def run(self):
        self.p_audio = pyaudio.PyAudio()
        while not self.stop_flag:
            if not self.play_queue.empty():
                item = self.play_queue.get()
                audio_file = item[0]
                try:
                    with wave.open(audio_file, 'rb') as wf:
                        if self.callback:
                            self.callback(item[1])
                        self.play_audio(wf)
                except Exception as e:
                    print(f"Error occurred while playing audio file: {audio_file}")
                    print(e)
                finally:
                    os.remove(audio_file)
            else:
                time.sleep(0.1)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.p_audio.terminate()
        print("Audio player stopped")

    def play_audio(self, audio_file):
        if not self.stream:
            self.stream = self.p_audio.open(
                format=self.p_audio.get_format_from_width(audio_file.getsampwidth()),
                channels=audio_file.getnchannels(),
                rate=audio_file.getframerate(),
                output=True
            )

        data = audio_file.readframes(1024)
        while data and not self.stop_flag:
            self.stream.write(data)
            data = audio_file.readframes(1024)
            while self.pause_flag:
                if not self.stop_flag:
                    print("Paused...")
                else:
                    print("Stopping...")
                time.sleep(1.0)

    def pause(self):
        self.pause_flag = not self.pause_flag

    def stop(self):
        self.stop_flag = True
        self.pause_flag = False
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--url', type=str)
    parser.add_argument('--xpath_config', type=str, default='xpath_config.json')

    args = parser.parse_args()
    txt_path = args.path
    web_url = args.url

    xpath_config = {}
    with open(args.xpath_config, 'r') as f:
        xpath_config = json.load(f)

    play_queue = queue.Queue()
    tts = TTSMelo(language='ZH', device='auto', speed=0.9)
    if txt_path:
        text_reader = TextReader(txt_path, play_queue, tts)
    elif web_url:
        text_reader = WebReader(web_url, play_queue, tts, xpath_config)
    else:
        raise ValueError("Either text file path or web url is required")    
    
    audio_player = AudioPlayer(play_queue)

    threads = [text_reader, audio_player]
    for thread in threads:
        thread.start()

    try:
        while True:
            time.sleep(0.5)
            if keyboard.is_pressed('space'):
                audio_player.pause()
                text_reader.pause()

    except KeyboardInterrupt:
        print("Stopping...")
        print("Audio player stopping takes a while...")
        for thread in threads:
            thread.stop()
        for thread in threads:
            thread.join(1.0)

        sys.exit(0)

if __name__ == "__main__":
    main()
    
#     chrome_options = Options()

# # Add the "headless" argument
#     chrome_options.add_argument("--headless")
#     driver = webdriver.Chrome()

#     # Go to a website that uses a JavaScript redirect
#     driver.get("https://www.shuke8.cc/frxxc/268573.html")

#     # The driver will wait for the page to fully load, including any JavaScript redirects
#     # You can then get the current URL, which will be the final URL after all redirects
#     final_url = driver.current_url
#     html_c=driver.page_source
#     print(html_c)

#     # Don't forget to close the driver when you're done with it
#     driver.quit()

#     print(final_url)
#     res = requests.get(final_url)
#     res.raise_for_status()

#     xpath = "//main[@class='container']/section/div[1]/h1" #"//a[@id='keyright']"#"//article[@id='article']"
#     content = res.content
#     content = html.fromstring(content)
#     content = content.xpath(xpath)
#     # content = ''.join([i.text_content() for i in content])
#     if content:
#         # href = content[0].attrib['href']
#         print(content[0].text_content())