from pathlib import Path
import sys,json,time,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,scipy
import experiment as e
from three_body import invariants
out=Path(__file__).resolve().parents[1]/'results/interactive_validation';out.mkdir(exist_ok=True,parents=True)
records=[];assertions=0

def check(condition,msg):
    global assertions
    assert condition,msg
    assertions+=1

# Κάθε σετ, και οι δύο διευθύνσεις, 5 μεγέθη: γεωμετρία, αναφορά, ορμή.
for key in e.PRESETS:
    reference=e.initial_pair(key,0)[0]
    for direction in ['x','y']:
        for fraction in [0,1e-8,1e-6,1e-2,.1]:
            a,b,meta=e.initial_pair(key,fraction,direction)
            m=e.PRESETS[key]['masses']
            check(np.array_equal(a,reference),f'A mutated {key}')
            check(np.array_equal(a[6:],b[6:]),'velocities changed')
            check(meta['com_difference']<1e-15,'COM shift')
            com=lambda y:np.sum(y[:6].reshape(3,2)*m[:,None],axis=0)/sum(m)
            check(np.linalg.norm(com(a)-com(b))<1e-14,'actual COM changed')
            check(np.isclose(meta['max_shift'],fraction*meta['characteristic_length']),'wrong fraction')
            check(np.allclose(invariants(a[None],m)[1],invariants(b[None],m)[1],atol=1e-15),'momentum changed')
            if fraction==0:check(np.array_equal(a,b),'zero perturbation')

for args in [('bad',1e-6,'x'),('euler',-.1,'x'),('euler',float('nan'),'x'),('euler',.2,'x'),('euler',1e-6,'z')]:
    try:e.initial_pair(*args)
    except ValueError:assertions+=1
    else:raise AssertionError('invalid input accepted')
for end in [0,-1,float('inf'),float('nan'),12.01]:
    try:e.compare('euler',end=end)
    except ValueError:assertions+=1
    else:raise AssertionError('invalid duration accepted')

configs=[(key,1e-6,'x',c['duration']) for key,c in e.PRESETS.items()]
configs += [('euler_long',0,'x',12),('euler_long',1e-8,'x',12),
 ('euler_long',1e-6,'y',12),('euler_long',.01,'x',12),('euler_long',.1,'y',12),
 ('lagrange_stable',1e-6,'x',60),('lagrange_equal',1e-6,'y',12),
 ('at_rest',1e-6,'x',3),('at_rest',0,'x',3),('burrau',1e-6,'y',15.8),
 ('burrau',.01,'x',20),('euler',1e-8,'x',.05)]
for i,(key,fraction,direction,end) in enumerate(configs):
    started=time.time();caseout=out/f'case_{i:02}'
    result=e.compare(key,fraction,direction,end,output=caseout)
    r=result['report'];r['elapsed_seconds']=time.time()-started
    check(np.isfinite(result['A']).all() and np.isfinite(result['B']).all(),'nonfinite result')
    check(np.all(np.diff(result['t'])>0),'time not monotone')
    check(r['actual_end']<=end,'extrapolation')
    check(r['com_difference']<1e-14,'COM')
    if fraction==0:
        check(np.array_equal(result['A'],result['B']),'identical initial state diverged')
        check(np.all(result['delta']==0),'nonzero D for identical state')
        check(r['final_amplification'] is None,'division by zero')
    if key=='at_rest' and end==3:check(r['stopped_early'],'collision not stopped')
    if key=='euler_long' and fraction==1e-6 and direction=='x':check(r['final_distance']>.5,'no visible divergence')
    if i in [0,1,7,9,14]:
        e.snapshot(result,caseout/'comparison.png')
        (caseout/'motion.html').write_text(e.animation_html(result,480))
    records.append(r)
    print(i,key,fraction,direction,end,'->',r['actual_end'],'D',r['final_distance'],'checks',r['numerical_checks_passed'],flush=True)
    (out/'validation.json').write_text(json.dumps(dict(assertions=assertions,records=records),ensure_ascii=False,indent=2))

# Πραγματικά callbacks widgets: αλλαγή, σύνδεση χρόνου, reset, εκτέλεση, ποιότητα, invalidation.
import IPython.display as ipd
original=ipd.display
ipd.display=lambda *a,**k:None
try:
    ui=e.show_experiment()
    ui['choice'].value='lagrange_stable'
    check(ui['duration'].max==60,'preset duration cap')
    ui['duration'].value=31.25
    check(ui['endfield'].value==31.25,'slider->number')
    ui['endfield'].value=30.5
    check(ui['duration'].value==30.5,'number->slider')
    ui['choice'].value='euler_long'
    check(ui['duration'].value==12 and ui['duration'].max==12,'preset reset')
    ui['amount'].value=-1
    check(not ui['epsilon'].disabled,'custom slider disabled')
    ui['epsilon'].value=.01;ui['direction'].value='y'
    ui['reset'].click()
    check(ui['amount'].value==1e-6 and ui['direction'].value=='x','reset failed')
    ui['run'].click()
    check(ui['state']['result'] is not None,'callback execution failed')
    check(ui['state']['result']['report']['numerical_checks_passed'],'callback numerical checks')
    check(ui['state']['html'] is not None,'automatic animation failed')
    for quality in [240,720,480]:
        ui['quality'].value=quality;ui['movie_button'].click()
        check(ui['state']['html'] is not None,'animation quality callback')
    ui['amount'].value=0
    check(ui['state']['result'] is None and ui['movie_button'].disabled,'stale result retained')
    ui['duration'].value=.05;ui['run'].click()
    check(ui['state']['result']['report']['final_distance']==0,'zero callback')
    # Τεχνητή αποτυχία μόνο για έλεγχο αποκατάστασης των χειριστηρίων.
    real=e.compare
    def fail(*a,**k):raise RuntimeError('controlled test failure')
    e.compare=fail;ui['run'].click();e.compare=real
    check(ui['state']['result'] is None and not ui['run'].disabled and not ui['choice'].disabled,'failure recovery')
    ui['run'].click()
    check(ui['state']['result'] is not None,'recovery next run failed')
finally:
    ipd.display=original
summary=dict(assertions=assertions,configuration_count=len(records),
    integrations=sum(3 if r['fraction']==0 else 6 for r in records),
    passed_numerical=sum(r['numerical_checks_passed'] for r in records),
    python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,records=records)
(out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('COMPLETE',assertions,'assertions',len(records),'configurations',flush=True)
