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
    refreshtoken = "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.CQ3YXusZmJJUZvGv8hhymb1_xYzE5M0xh-bf76_pAhKbmCma6-l2255gxrKN31gKUZvEryQ81hQL1EH05-KRT8wUz8vIezG-mspj_KWsR9ZCXYHYWO0ByWQmATlbD2L37VrWU2lxtSLnincin-o0UAmgWJ9OQAbJKqTVtH328l8MM1QcbzsSMYkNEJG3cVT90W7URT0KGK06hc6V-VCQljKfyraSfTFHSrSt0uWX4HL8aHzmv0NyNYeg_vHzDo8QM2XGeIBizfdf0CGin7Y6JMeJoSo9ORO6tM4UjsvB5QDn1TAeUHdnjliNQmDSWMcdqXuwUm_wyzTzvxy98Mt0uQ.92uyonyOPP9VR13e._r9wRnlfLFq2jUg-vDEEIVqv-KKUqBppLw8IYIbKvaJ5DWBf5_OZ98Aw8Cyfo4Go1dPTvsA4Ya4FhoZMy5TI-oSMIVAv7R1IQ59nIQ5mhqaHEHCjwxi8lQE48Tl9gKhqKP10X0OhpBgQc096DoHq8BCHisM61MRcFPu-3LJ7g0gv1t0b4RrvwBsveqc8SfRuwVunQt6itKEJ_bmo_6SWE6YxQ93Hj8Ex9p5GgG1r00npeIyKeNhjDJpuxvob3dyhbIhZ0itjPWYZ6tz5KbuRd1TiDkG3M_5isO3_KHAY-AJMe5_JlqaCfuBEaBwxKS5__z2VMeFE5J1CwiSfjlX6711Tnuy8DU3pD4MVPZQcB9BrsSvfPVqwx41gpqMF0ebk-P1ObwSXiDQYpe4WB2svZpexFAWTcjO9ScldHUfPS2MRVfBED8eDrTr9GFsbfux_RRTGEmo5UFWtjnp3Q6fSybcfuc0a16kOCPRt3ZpAC53aedQBYLXgRPqJAyzPyYZtocmXZUxGMgPZj3lpi387rq33Ql-9yzwF9yPXTeg0oUmSp79m-qcH11Ng_mOT1Pl-Z-KsZ-R2E6thwYhPY45JsXy_CmyoGgkhiEHvU-M7ThYbirq6PiQ2kIOeFHzRKgP2xAj3VwBj4gkCIl8fJmBYFL9IqCLahUDF1GzXxd0GI6MHpl_mu1V6O_-3bpY_Ukq3XFFykkavrwezw8SEGBDVuemGffMmd3XHjHi58IXt2_e-rsTgOHqiXv4NHIppqzeBttbLBZeOJnoqZ6R8wd-ZFulPoaXmYYTUl3R_Dh7ItmRHgNf_8w5I5UbpEDzh8iOrqgqwGtt1gGZOIEMfa29PD9LrGZFnSGUk16mUpjx7WJawmmAe0xU9KOw-5Zn3el6SmodiXp1AVFEwlPPWfSOIRGUXCYjIr7VSnMh5crGx-2fOQDHJozv2c7sJJSQvOSSpJJW2q4OXlpoQWCQcD9cAbywe8ZLnR4j9DAE1oIclRjPa_nBdrm9YhrP3hqLJFW7BX2M5WygWBeoZShIfJ5Lpsn501ts1jvooXPB2QaCi8fLxq3UtvruZ2JmZx_Ri35_mR2J8x6kutqW6WRta-4CFhpDyN7iohILhOLtPNbRCeP8QPsTxeqHdFIKip5T3Nw5OxWcPbOB0LCXr94kwgy452DTdjk4Bp1S6NZFPCGWp2-1dJex4jH-_neKOXDxztDr_9uK5xPeTn_ajQFWj7zd_RPYnxCsJ_wTSwX13kRz7AmZUnC_ION3pEEbXlZCnsYHiT789ozRIOpE0TMWq3A6t6AlC4FAaA6OlgcZomMvxH4bGOahG92iggiR4OcnHtzFwzKcXFNX-bsnwPA.P6dIhEmmt-8w22t69CD44A"#@param {type: "string"}

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