"""Frankfurter v2 currency-only replacement behind the native raw-JSON seam.

Manifest url/host continue to match the pinned upstream invocation; they are NOT
sent to Frankfurter. This version fixes the actual no-key endpoint below. Preserve
full float multiplication (no rounding); the native tool still returns float.
"""
from datetime import date
import math
import re

BACKEND_VERSION = 'frankfurter-v2-rate-v1'
ENDPOINT_TEMPLATE = 'https://api.frankfurter.dev/v2/rate/{base}/{quote}'


def request_parameters(arguments):
    if set(arguments) != {'from','to','amount'}:
        raise ValueError('currency request arguments mismatch')
    for key in ('from','to'):
        if type(arguments[key]) is not str or re.fullmatch('[A-Z]{3}',arguments[key]) is None:
            raise ValueError('currency codes must be native validated uppercase codes')
    amount = arguments['amount']
    if type(amount) not in (int,float) or not math.isfinite(amount):
        raise ValueError('finite currency amount required')
    return dict(url=ENDPOINT_TEMPLATE.format(base=arguments['from'],quote=arguments['to']),
                params={},headers={})


def adapt_response(body, arguments):
    if type(body) is not dict or set(body) != {'date','base','quote','rate'}:
        raise ValueError('Frankfurter response shape mismatch')
    if body['base'] != arguments['from'] or body['quote'] != arguments['to']:
        raise ValueError('Frankfurter currency pair mismatch')
    if type(body['date']) is not str or date.fromisoformat(body['date']).isoformat() != body['date']:
        raise ValueError('Frankfurter rate date invalid')
    rate = body['rate']
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Frankfurter positive finite rate required')
    amount = float(arguments['amount']) * rate
    if not math.isfinite(amount):
        raise ValueError('converted amount overflow')
    return {'result':{'convertedAmount':amount}, 'provider_response':dict(body),
            'provider_backend_version':BACKEND_VERSION}
