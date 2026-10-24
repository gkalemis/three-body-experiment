from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as e
namespace=dict(e=e,assertions=0)
exec('''def check(value,msg):
 global assertions
 assert value,msg
 assertions+=1
''',namespace)
source=(Path(__file__).with_name('validate_interactive.py')).read_text()
exec(source.split('# Πραγματικά callbacks widgets:')[1].split('\n',1)[1].split('summary=dict(')[0],namespace)
ui=namespace['ui'];check=namespace['check']
import types
colab=types.ModuleType('google.colab');files=types.SimpleNamespace(download=lambda p:downloads.append(p));colab.files=files
downloads=[]
sys.modules['google.colab']=colab
ui['download'].click()
check(len(downloads)==1 and downloads[0].endswith('motion.html'),'Colab download dispatch')
ui['quality'].value=240
ui['download'].click()
check(len(downloads)==2,'rebuild before download')
ui['direction'].value='y'
check(ui['download'].disabled,'stale download enabled')
print('WIDGETS PASSED',namespace['assertions'])
