from datetime import timedelta
import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('MY_API_KEY')
IDs = ["11B00000", "11B10101"]


# 기상청 허브를 사용하여 육상예보 및 기온예보를 받아오는 함수
def get_information(start_date, end_date, url, type):
    # 지상 예보 호출용 REG_ID
    if type == 0:
        id = IDs[0]
    # 기온 예보 호출용 REG_ID
    else:
        id = IDs[1]

    # 전체 데이터를 저장할 dictionary 리스트
    entire_data = []

    # 시작 날짜부터 종료 날짜까지 순차적으로 호출하기 위한 변수
    current_date = start_date

    while current_date < end_date:

        # 날짜 간격 계산 (최대 10일)
        temp_end_date = min(current_date + timedelta(days=9), end_date)

        # 날짜 포매팅
        formatted_current_date = current_date.strftime("%Y%m%d%H%M")
        formatted_temp_end_date = temp_end_date.strftime("%Y%m%d%H%M")

        params = {
            "reg": id,
            "tmfc1": formatted_current_date,
            "tmfc2": formatted_temp_end_date,
            "tmef1": formatted_current_date[:8],  # 연월일만 사용
            "tmef2": formatted_temp_end_date[:8],  # 연월일만 사용
            "mode": "0",
            "disp": "0",
            "help": "0",
            "authKey": api_key
        }

        # GET 요청 보내기
        response = requests.get(url, params)

        # 응답 결과의 문자열을 공백을 지운 후 줄 바꿈을 기준으로 리스트로 저장함
        lines = response.text.strip().split('\n')

        # 열 이름을 저장할 리스트
        header_line = []

        # dictionary 사용하여 저장
        for line in lines:
            # 헤더를 따로 구분함
            if "REG_ID" in line:
                line = line[1:]
                header_line = line.split()
                continue
            # 도움말 부분을 스킵함
            if line.startswith("#") or not line.strip():
                continue

            # [REG_ID, TM_FC, TM_EF, ect...]
            parts = line.split()

            # {"REG_ID" : REG_ID, "TM_FC" : TM_FC, "TM_EF" : TM_EF, ...}
            record = {header_line[j]: parts[j] for j in range(len(header_line))}

            # [{"REG_ID" : REG_ID, "TM_FC" : TM_FC, "TM_EF" : TM_EF, ...}, {"REG_ID" : REG_ID, "TM_FC" : TM_FC, "TM_EF" : TM_EF, ...}, ...]

            entire_data.append(record)

        # 다음 반복을 위해 시작 날짜 업데이트
        current_date = temp_end_date + timedelta(minutes=1)  # 다음 날짜로 진행

    return entire_data
