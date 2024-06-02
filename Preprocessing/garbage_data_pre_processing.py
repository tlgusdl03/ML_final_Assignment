import pandas as pd
import os


# 모든 쓰레기 배출량 데이터를 합치는 함수
def concat_data():

    # 현재 스크립트 파일의 절대 경로
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 데이터가 저장된 디렉토리 경로
    target_path = os.path.join(current_path, "garbage_data")

    # target_path에서 .csv 파일만 찾아서 전체 경로 리스트 생성
    csv_path_list = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith(".csv")]

    # 모든 CSV 파일을 데이터프레임 리스트로 읽기
    dataframes = [pd.read_csv(filepath) for filepath in csv_path_list]

    # 모든 데이터프레임을 하나로 합치기
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df
    # # 결과를 CSV 파일로 저장
    # output_path = os.path.join(current_path, "combined_data.csv")
    # combined_df.to_csv(output_path, index=False)  # index=False로 설정하면 인덱스는 저장하지 않음
    #
    # print(f"데이터가 {output_path} 에 저장되었습니다.")

# 서울특별시의 데이터만 추출하는 함수
def extract_data(combined_df):

    # 'city_do_nm' 열에서 '서울특별시'와 일치하는 행만 필터링
    extracted_df = combined_df[combined_df['city_do_nm'] == '서울특별시']

    return extracted_df


# 구별로 나뉜 쓰레기 배출량 데이터를 서울특별시로 통합한 함수
def aggregate_dscamt_by_district(seoul_df):
    pd.options.mode.copy_on_write = True

    seoul_df.loc[:, 'date'] = pd.to_datetime({
        'year': seoul_df['year'],
        'month': seoul_df['mt'],
        'day': seoul_df['dt']
    })

    seoul_df.drop(['year', 'mt', 'dt'], axis=1, inplace=True)
    # 'city_gn_gu_nm'으로 그룹화하고 'dscamt' 열의 값을 합산
    aggregated_data = seoul_df.groupby(['city_do_nm', 'date'])['dscamt'].sum()

    # 결과를 CSV 파일로 저장
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_path, "garbage_combined_data.csv")

    aggregated_data.to_csv(output_path)  # index=False로 설정하면 인덱스는 저장하지 않음

    print(f"데이터가 {output_path} 에 저장되었습니다.")

    return aggregated_data
