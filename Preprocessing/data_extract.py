import pandas as pd


# 기상 데이터에서 중복된 날의 데이터를 제거하는 함수
def data_extract(entire_data):
    unique_days = {}
    for record in entire_data:
        ef_date = record['TM_EF'][:8]
        if ef_date not in unique_days:
            unique_days[ef_date] = record
    return unique_days


# 기상예보 데이터와 기온예보 데이터를 통합하는 함수
def data_merge(ground_data, temperature_data):

    merged_data = []

    ground_list = list(ground_data.items())
    temperature_list = list(temperature_data.items())

    # 두 데이터 세트를 zip을 사용하여 동시에 순회합니다.
    for (g_date, ground), (t_date, temperature) in zip(ground_list, temperature_list):
        # 'TM_EF' 필드의 년월일 부분만 비교
        if ground['TM_EF'] == temperature['TM_EF']:
            merged = {**ground, **temperature}
            merged_data.append(merged)

    merged_data_df = pd.DataFrame(merged_data)

    merged_data_df['date'] = pd.to_datetime(merged_data_df['TM_EF'], format='%Y%m%d%H%M')

    return merged_data_df

# 데이터를 csv 파일로 내보내는 함수
def data_write(merged_data, filename):
    merged_data.to_csv(filename)