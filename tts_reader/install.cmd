pip install -r requirements.txt &&
git clone https://github.com/myshell-ai/MeloTTS.git &&
cd MeloTTS &&
pip install -e . &&
python -m unidic download &&
cd ..