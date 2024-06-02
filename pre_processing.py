from datetime import datetime, timedelta
from get_information import get_information

# URL 정의
ground_weather_url = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
temperature_url = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wc.php"

# datetime 객체 생성 및 형식 조정
start_date = datetime(2021, 6, 1, 0, 0)
end_date = datetime(2021, 6, 20, 23, 30)
# formatted_start_date = start_date.strftime("%Y%m%d%H%M")
# formatted_end_date = end_date.strftime("%Y%m%d%H%M")

# 정보 받아옴
get_information(start_date, end_date, ground_weather_url, 0)
get_information(start_date, end_date, temperature_url, 1)