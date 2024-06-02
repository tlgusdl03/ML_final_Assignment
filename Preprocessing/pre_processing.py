from datetime import datetime
import pandas as pd
from Preprocessing.data_extract import data_extract, data_merge, data_write
from Preprocessing.garbage_data_pre_processing import concat_data, extract_data, aggregate_dscamt_by_district
from Preprocessing.get_information import get_information

sky_mapping = {
    'WB01': 0,
    'WB02': 1,
    'WB03': 2,
    'WB04': 3
}

pre_mapping = {
    'WB00': 0,
    'WB09': 1,
    'WB11': 2,
    'WB12': 3,
    'WB13': 4
}

wf_mapping = {
    '"맑음"' : 0,
    '"구름많음"' : 1,
    '"구름많고' : 1,
    '"흐리고' : 2,
    '"흐림"' : 2
}

rn_st_mapping = {
    '비/눈"' : 100,
    '비/"' : 100,
    '비"' : 100
}

#URL 정의
ground_weather_url = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
temperature_url = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wc.php"

# datetime 객체 생성 및 형식 조정
start_date = datetime(2019, 12, 1, 0, 0)
end_date = datetime(2021, 7, 31, 23, 30)
formatted_start_date = start_date.strftime("%Y%m%d%H%M")
formatted_end_date = end_date.strftime("%Y%m%d%H%M")

# 정보 받아옴
ground_data = get_information(start_date, end_date, ground_weather_url, 0)
temperature_data = get_information(start_date, end_date, temperature_url, 1)

# 시계열 데이터를 일별로 정렬시킴
arranged_ground_data = data_extract(ground_data)
arranged_temperature_data = data_extract(temperature_data)

# print(arranged_ground_data)
# print(arranged_temperature_data)

# 데이터를 합침
merged_data = data_merge(arranged_ground_data, arranged_temperature_data)

# 데이터를 파일로 출력함
data_write(merged_data, 'merged_data.csv')

concatenated_data = concat_data()

extracted_data = extract_data(concatenated_data)

aggregated_data = aggregate_dscamt_by_district(extracted_data)

total_data = pd.merge(aggregated_data, merged_data, on='date', how='outer')

total_data.to_csv('total_data.csv')

print(total_data)

total_data = pd.read_csv("/total_data.csv")

total_data['date'] = pd.to_datetime(total_data['date'])

total_data.set_index('date', inplace=True)  # 날짜를 인덱스로 설정

total_data = total_data.infer_objects()

for col in total_data.columns:
    if total_data[col].dtype == 'float64' or total_data[col].dtype == 'int64':
        total_data[col] = total_data[col].interpolate(method='linear')
    elif total_data[col].dtype == 'object':
        total_data[col] = total_data[col].fillna(method='ffill')

print(total_data.head())
print(total_data.dtypes)
print(total_data.isnull().sum())

total_data.drop(['REG_ID', 'TM_FC', 'TM_EF', 'MOD', 'STN', 'C', 'CONF'], axis=1, inplace=True)

total_data['SKY'] = total_data['SKY'].map(sky_mapping)
total_data['PRE'] = total_data['PRE'].map(pre_mapping)
total_data['WF'] = total_data['WF'].map(wf_mapping)
total_data['RN_ST'] = total_data['RN_ST'].map(rn_st_mapping).fillna(total_data['RN_ST'])

total_data.to_csv("after_data.csv")