python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ipykernel
pip install -r requirements.txt
pip list
python3 -m ipykernel install --user --name=StockAnalysisTest
deactivate