import pandas as pd
import ast

df = pd.read_json('data.json')


def parse_dict_field(value):
	"""Parse values that may be dicts, stringified dicts, or nulls."""
	if isinstance(value, dict):
		return value

	if isinstance(value, str):
		value = value.strip()
		if not value:
			return {}
		try:
			parsed = ast.literal_eval(value)
			if isinstance(parsed, dict):
				return parsed
		except (ValueError, SyntaxError):
			return {}

	return {}


transformed_df = df.copy()

address_index = transformed_df.columns.get_loc('address')
address_parsed = transformed_df.pop('address').apply(parse_dict_field)

transformed_df.insert(address_index, 'address_country_code', address_parsed.apply(lambda x: x.get('country_code')))
transformed_df.insert(address_index + 1, 'address_latitude', address_parsed.apply(lambda x: x.get('latitude')))
transformed_df.insert(address_index + 2, 'address_longitude', address_parsed.apply(lambda x: x.get('longitude')))
transformed_df.insert(address_index + 3, 'address_region_name', address_parsed.apply(lambda x: x.get('region_name')))
transformed_df.insert(address_index + 4, 'address_town', address_parsed.apply(lambda x: x.get('town')))

naics_index = transformed_df.columns.get_loc('primary_naics')
primary_naics_parsed = transformed_df.pop('primary_naics').apply(parse_dict_field)

transformed_df.insert(naics_index, 'primary_naics_code', primary_naics_parsed.apply(lambda x: x.get('code')))
transformed_df.insert(naics_index + 1, 'primary_naics_label', primary_naics_parsed.apply(lambda x: x.get('label')))

print(transformed_df.head(10))

transformed_df.to_json('transformed_data.json', orient='records', indent=2)