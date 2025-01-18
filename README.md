git clone https://github.com/hardware-is-not-software/daytrader.git

// Install Python and preferably virtual enviroment as miniconda

pip install -r requirements.txt

python daytrader.py --help

# Single stock
python daytrader.py --ticker INTC --stoploss -15 --tradecost 0.2

# List of stocks (eg. index)
python daytrader.py --stocklist lists/SP5.csv --tradecost 0.2
