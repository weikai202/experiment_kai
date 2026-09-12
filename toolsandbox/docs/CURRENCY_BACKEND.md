# Currency backend replacement

The user authorized replacing the retired Currency Converter 18 backend. The public
`convert_currency(amount, from_currency_code, to_currency_code) -> float` contract
and native currency-code validation remain unchanged. The project intercepts the
internal HTTP boundary; the installed, pinned ToolSandbox dependency is read-only.

New live runs use Frankfurter v2 (`https://api.frankfurter.dev/v2/rate/{base}/{quote}`),
which requires no account or API key. It publishes daily reference rates, not
intraday trading quotes. Conversion multiplies the requested amount by the returned
rate without introducing monetary rounding. Unsupported currencies and HTTP errors
remain failures; there is no fabricated rate or alternate-provider fallback.

Official documentation: https://frankfurter.dev/

The backend version and manifest hash distinguish new requests and recordings from
Currency Converter 18 fixtures. Legacy recordings remain identifiable as legacy
recordings. The upstream URL retained in an interception manifest is a matching
identity for the pinned function's request, not the actual new HTTP destination.
The adapter version fixes the actual Frankfurter endpoint. RapidAPI credentials
must never be requested or transmitted for currency conversion.

Other external tools (weather, geocoding, maps, and stocks) retain their existing
backends and configuration requirements. Replacing currency conversion alone does
not establish full-experiment readiness.

Validation on this instance:
- Three direct native `convert_currency` calls succeeded through the new boundary.
- Five native external tools returned HTTP 200 and nonempty parsed results.
- Reports: `/root/toolsandbox-runtime/currency-backend-smoke.json` and
  `/root/toolsandbox-runtime/external-preflight-result.json`.
- The new manifest is `configs/reproducibility/rapidapi_backends_frankfurter_v2.json`,
  SHA-256 `b64a3f53388444fde4e17834fad12b1cbbac58d3afb43cecf289f011688193a4`.
- The adapter is `src/toolsandbox_pipeline/toolsandbox_adapter/currency_backend.py`.
