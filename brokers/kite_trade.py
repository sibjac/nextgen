import json
import os

from dotenv import load_dotenv
from urllib3 import Retry

from brokers.utils import login
try:
    import requests
except ImportError:
    os.system('python -m pip install requests')
try:
    import dateutil
except ImportError:
    os.system('python -m pip install python-dateutil')


import requests
import dateutil.parser
from requests.adapters import HTTPAdapter
load_dotenv()



class ZerodhaBroker:
    # Products
    PRODUCT_MIS = "MIS"
    PRODUCT_CNC = "CNC"
    PRODUCT_NRML = "NRML"
    PRODUCT_CO = "CO"

    # Order types
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SLM = "SL-M"
    ORDER_TYPE_SL = "SL"

    # Varities
    VARIETY_REGULAR = "regular"
    VARIETY_CO = "co"
    VARIETY_AMO = "amo"

    # Transaction type
    TRANSACTION_TYPE_BUY = "BUY"
    TRANSACTION_TYPE_SELL = "SELL"

    # Validity
    VALIDITY_DAY = "DAY"
    VALIDITY_IOC = "IOC"

    # Exchanges
    EXCHANGE_NSE = "NSE"
    EXCHANGE_BSE = "BSE"
    EXCHANGE_NFO = "NFO"
    EXCHANGE_CDS = "CDS"
    EXCHANGE_BFO = "BFO"
    EXCHANGE_MCX = "MCX"

    def __init__(self):
        
        # Define your retry strategy
        retry_strategy = Retry(
            total=5,                              # total number of retries
            status_forcelist=[429, 500, 502, 503, 504],  # retry on these HTTP status codes
            backoff_factor=1,                     # exponential backoff: 1s, 2s, 4s, etc.
            raise_on_status=False,                # don't raise exceptions for retriable status codes
        )

        # Mount the retry strategy to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        self.enctoken = login()["enctoken"]
        self.headers = {"Authorization": f"enctoken {self.enctoken}"}
        self.session = requests.session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.root_url = "https://kite.zerodha.com/oms"
        self.session.get(self.root_url, headers=self.headers)
        


    def instruments(self, exchange=None):
        data = self.session.get(f"https://api.kite.trade/instruments").text.split("\n")
        Exchange = []
        print(data)
        for i in data[1:-1]:
            row = i.split(",")
            if exchange is None or exchange == row[11]:
                Exchange.append({'instrument_token': int(row[0]), 'exchange_token': row[1], 'tradingsymbol': row[2],
                                 'name': row[3][1:-1], 'last_price': float(row[4]),
                                 'expiry': dateutil.parser.parse(row[5]).date() if row[5] != "" else None,
                                 'strike': float(row[6]), 'tick_size': float(row[7]), 'lot_size': int(row[8]),
                                 'instrument_type': row[9], 'segment': row[10],
                                 'exchange': row[11]})
        return Exchange

    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
        params = {"from": from_date,
                  "to": to_date,
                  "interval": interval,
                  "continuous": 1 if continuous else 0,
                  "oi": 1 if oi else 0}
        

        lst = self.session.get(
            f"{self.root_url}/instruments/historical/{instrument_token}/{interval}", params=params,
            headers=self.headers)
        



        lst = lst.json()["data"]["candles"]
        records = []
        for i in lst:
            record = {"date": dateutil.parser.parse(i[0]), "open": i[1], "high": i[2], "low": i[3],
                      "close": i[4], "volume": i[5],}

            if len(i) == 7:
                record["oi"] = i[6]
            records.append(record)
        return records

    def quote(self, instrument : str):
        
        #  params={ "i": instrument 
        quotes = self.session.get(f"https://api.kite.trade/quote", headers=self.headers )  .json()
        return quotes

    
    def margins(self):
        margins = self.session.get(f"{self.root_url}/user/margins", headers=self.headers).json()["data"]
        return margins

    def profile(self):
        profile = self.session.get(f"{self.root_url}/user/profile", headers=self.headers).json()["data"]
        return profile

    def orders(self):
        orders = self.session.get(f"{self.root_url}/orders", headers=self.headers).json()["data"]
        return orders

    def positions(self):
        positions = self.session.get(f"{self.root_url}/portfolio/positions", headers=self.headers).json()["data"]
        return positions

    def place_order(self, variety, exchange, tradingsymbol, transaction_type, quantity, product, order_type, price=None,
                    validity=None, disclosed_quantity=None, trigger_price=None, squareoff=None, stoploss=None,
                    trailing_stoploss=None, tag=None):
        params = locals()
        del params["self"]
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]
        order_id = self.session.post(f"{self.root_url}/orders/{variety}",
                                     data=params, headers=self.headers).json()["data"]["order_id"]
        return order_id

    def modify_order(self, variety, order_id, parent_order_id=None, quantity=None, price=None, order_type=None,
                     trigger_price=None, validity=None, disclosed_quantity=None):
        params = locals()
        del params["self"]
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        order_id = self.session.put(f"{self.root_url}/orders/{variety}/{order_id}",
                                    data=params, headers=self.headers).json()["data"][
            "order_id"]
        return order_id

    def cancel_order(self, variety, order_id, parent_order_id=None):
        order_id = self.session.delete(f"{self.root_url}/orders/{variety}/{order_id}",
                                       data={"parent_order_id": parent_order_id} if parent_order_id else {},
                                       headers=self.headers).json()["data"]["order_id"]
        return order_id

    def close(self):
        # No-op for compatibility with BaseBroker interface
        pass
