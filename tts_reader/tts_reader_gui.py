import os
import json
import tkinter as tk 
import queue

from tkinter import ttk, messagebox
from tkinter import scrolledtext, Button, Entry
from tts_reader import TTSMelo, TextReader, WebReader, AudioPlayer

LABEL_TEXT = """
TTS Book Reader, can read local text files or text from the internet.
Curently only supports English and Chinese.
Curently only supports website from www.shuke8.cc, e.g. https://www.shuke8.cc/tsxk/287095.html
"""

XPATH_CONFIG_PATH = 'xpath_config.json'
xpath_config = None
tts = None
threads = None
cur_reading_book = None

def set_text_area(text_area, text):
    text_area.insert(tk.END, text)
    text_area.see('end')

def read_book(event, book, text_area):
    global threads, cur_reading_book
    read_btn = event.widget
    cur_book_input = book.get().strip().lstrip('"').rstrip('"')
    if cur_reading_book == cur_book_input:
        if threads:
            for thread in threads:
                thread.pause()
                if thread.pause_flag:
                    read_btn.config(text='Read')
                    book.config(state='normal')
                else:
                    read_btn.config(text='Pause')
                    book.config(state='disabled')
        else:
            messagebox.showinfo("Book Reader", "Pleae input a valid book name or URL")
            read_btn.config(text='Read')
            book.config(state='normal')
        return
    
    read_btn.config(state='disabled')
    book.config(state='disabled')
    start_read = False
    try:
        cur_reading_book = cur_book_input
        if threads:
            for thread in threads:
                thread.stop()
            for thread in threads:
                thread.join()
            threads = None

        if os.path.exists(cur_reading_book) \
            and os.path.isfile(cur_reading_book) \
            and cur_reading_book.endswith('.txt'):
            play_queue = queue.Queue()
            text_reader = TextReader(cur_reading_book, play_queue, tts)
        elif cur_reading_book.startswith('http'):
            play_queue = queue.Queue()
            try:
                text_reader = WebReader(cur_reading_book, play_queue, tts, xpath_config)
            except Exception as e:
                messagebox.showerror("Book Reader", 
                    f"Error reading book from website\n'{cur_reading_book}'\n{e}")
                return
        else:
            messagebox.showerror("Book Reader", 
                f"The input is not a valid local file or website url\n'{cur_book_input}'")
            return
        
        audio_player = AudioPlayer(play_queue, callback=lambda text: set_text_area(text_area, text))

        threads = [text_reader, audio_player]
        print('Start reading book')
        for thread in threads:
            thread.start()
        read_btn.config(text='Pause')
        start_read = True
    except Exception as e:
        read_btn.config(text='Read')
        messagebox.showerror("Book Reader", f"Error reading book\n{e}")
    finally:
        read_btn.config(state='normal')
        if start_read:
            book.config(state='disabled')
        else:
            book.config(state='normal')

def on_close(win):
    global threads
    if threads:
        for thread in threads:
            thread.stop()
        for thread in threads:
            thread.join(1.0)
        threads = None
    win.destroy()

def main():
    # Creating tkinter main window 
    win = tk.Tk() 
    win.title("Book Reader")
    win.grid_columnconfigure(0, weight=1)
    for i in range(3):
        win.grid_rowconfigure(i, weight=1)

    win.resizable(False, False)
    win.protocol("WM_DELETE_WINDOW", lambda: on_close(win))
    
    # Title Label 
    label = ttk.Label(win,  
            text = LABEL_TEXT, 
            font = ("Times New Roman", 15))
    label.grid(row=0, column=0, sticky='ew') 
    
    # Frame for Book and Read Button
    frame = tk.Frame(win)
    frame.grid(row=1, column=0, sticky='ew')
    frame.grid_columnconfigure(0, weight=1)

    book = Entry(frame)
    book.grid(row=0, column=0, sticky='ew')
    book.insert(0, 'Enter book name or URL, then press Read button.')

    read_btn = Button(frame, text = 'Read', width=15) 
    read_btn.grid(row=0, column=1, sticky='e')
    # Creating scrolled text  
    # area widget 
    text_area = scrolledtext.ScrolledText(win,  
                                        wrap = tk.WORD, 
                                        font = ("Times New Roman", 
                                                15))
    text_area.grid(column=0, row=2, sticky='nsew')
    text_area.grid(column = 0, pady = 10, padx = 10) 

    read_btn.bind("<Button-1>", lambda event: read_book(event, book, text_area))

    book.focus()
    win.mainloop()


if __name__ == "__main__":
    xpath_config = json.load(open(XPATH_CONFIG_PATH, 'r'))
    tts = TTSMelo(language='ZH', device='auto', speed=0.9)
    main()