import json

values_path = 'task3/values.json'
tests_path = 'task3/tests.json'
report_path = 'task3/report.json'

def fill_values(tests_data, values_map):
    if isinstance(tests_data, dict):
        if 'id' in tests_data and tests_data['id'] in values_map:
            tests_data['value'] = values_map[tests_data['id']]
        if 'values' in tests_data:
            for item in tests_data['values']:
                fill_values(item, values_map)
    elif isinstance(tests_data, list):
        for item in tests_data:
            fill_values(item, values_map)
with open(values_path, 'r') as f:
    values_data = json.load(f)
with open(tests_path, 'r') as f:
    tests_data = json.load(f)
values_map = {item['id']: item['value'] for item in values_data['values']}
fill_values(tests_data, values_map)
with open(report_path, 'w') as f:
    json.dump(tests_data, f, indent=2)

