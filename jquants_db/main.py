import os
import sys
import json
import requests
import pandas as pd
from sqlalchemy import create_engine
from time import sleep
from IPython.display import display

pd.set_option("display.max_columns", None)
API_URL = "https://api.jquants.com"

def main():
    refreshtoken = os.getenv('REFRESH_TOKEN') #@param {type: "string"}

    # idToken取得
    res = requests.post(f"{API_URL}/v1/token/auth_refresh?refreshtoken={refreshtoken}")
    if res.status_code == 200:
        id_token = res.json()['idToken']
        headers = {'Authorization': 'Bearer {}'.format(id_token)}
        display("idTokenの取得に成功しました。")
    else:
        display(res.json()["message"])


    mailaddress=os.getenv("JQUANTS_MAILADDRESS")
    password=os.getenv("JQUANTS_PASSWORD")

    USER_DATA = {'mailaddress': mailaddress, 'password':password}

    # refresh token取得
    try:
        res = requests.post(f"{API_URL}/v1/token/auth_user", data=json.dumps(USER_DATA))
        refresh_token = res.json()['refreshToken']
    except:
        print("RefreshTokenの取得に失敗しました。")
    else:
        # id token取得
        try:
            res = requests.post(f"{API_URL}/v1/token/auth_refresh?refreshtoken={refresh_token}")
            id_token = res.json()['idToken']
        except:
            print("idTokenの取得に失敗しました。")
        else:
            headers = {'Authorization': 'Bearer {}'.format(id_token)}
            print("API使用の準備が完了しました。")
    
    code = ""#@param {type:"string"}
    date = "2025-06-13"#@param {type:"string"}

    params = {}
    if code != "":
        params["code"] = code
    if date != "":
        params["date"] = date

    res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=headers)
    if res.status_code == 200:
        d = res.json()
        data = d["info"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=headers)
            d = res.json()
            data += d["info"]
        df = pd.DataFrame(data)
        display(df)
    else:
        print(res.json())

    res = requests.get(f"{API_URL}/v1/fins/statements", params=params, headers=headers)
    if res.status_code == 200:
        d = res.json()
        data = d["statements"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(f"{API_URL}/v1/fins/statements", params=params, headers=headers)
            d = res.json()
            data += d["statements"]
        df = pd.DataFrame(data)
        display(df)
    else:
        print(res.json())

if __name__ == "__main__":
    main()