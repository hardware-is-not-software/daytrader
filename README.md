git clone https://github.com/hardware-is-not-software/daytrader.git

// Install Python and preferably virtual enviroment as miniconda

pip install -r requirements.txt

python daytrader.py --help

python daytrader.py --stock INTC --stoploss -15 --tradecost 0.2