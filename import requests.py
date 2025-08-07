import requests

def ocr_space_file(filename, api_key='K84773500388957'):
    payload = {'isOverlayRequired': False}
    with open(filename, 'rb') as f:
        r = requests.post(
            'https://api.ocr.space/parse/image',
            files={filename: f},
            data=payload,
            headers={'apikey': api_key}
        )
    result = r.json()
    return result['ParsedResults'][0]['ParsedText']