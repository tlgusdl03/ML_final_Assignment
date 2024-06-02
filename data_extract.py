from datetime import datetime


def data_extract(entire_data):
    unique_days = {}

    for data in entire_data:
        for record in data:
            ef_date = datetime.strptime(record['TM_EF'], '%Y%m%d%H%M').date()

            if ef_date not in unique_days:
                unique_days[ef_date] = record

