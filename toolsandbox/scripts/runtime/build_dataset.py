import locale,traceback
from pathlib import Path
locale.setlocale(locale.LC_ALL,'C.UTF-8')
from toolsandbox_pipeline.reproducibility.dataset_manifest import build_native_bundle
try:
 print(build_native_bundle(Path('/root/toolsandbox/configs/reproducibility/dataset_build_v1.json'),Path('/root/toolsandbox-runtime/dataset'),Path('/root/toolsandbox')))
except Exception as e:
 print(type(e).__name__,str(e))
 tb=e.__traceback__
 while tb:
  f=tb.tb_frame
  print(f.f_code.co_filename,tb.tb_lineno,f.f_code.co_name)
  if f.f_code.co_name=='_json':
   x=f.f_locals['value'];print('value type',type(x).__module__,type(x).__qualname__)
  if f.f_code.co_name=='_callable_identity':
   x=f.f_locals['function'];print('callable type',type(x).__name__,'identity',getattr(x,'__module__',None),getattr(x,'__qualname__',None),'closure',bool(getattr(x,'__closure__',None)))
  tb=tb.tb_next
 raise SystemExit(1)
