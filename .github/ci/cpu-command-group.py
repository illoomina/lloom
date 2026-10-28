#!/usr/bin/env python3
"""CPU CI command custody: retain PGID until descendants stop and are reaped.
Reuses the retained-group lifecycle; no scheduling, devices or test oracle.
"""
import ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parent
REPO=Path.cwd()
ENV=dict(os.environ)
SIGNALS=(signal.SIGTERM,signal.SIGINT,signal.SIGQUIT,signal.SIGHUP)
ADOPTED_REAPS=[]
class Stop(RuntimeError):pass


def no_duplicates(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise Stop('duplicate JSON key: '+k)
  d[k]=v
 return d


def read(path):
 p=Path(path)
 if p.stat().st_size>8*1024*1024:raise Stop('oversize receipt: '+str(p))
 return json.loads(p.read_text(),object_pairs_hook=no_duplicates)


def new_json(path,value):
 with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def require(x,message):
 if not x:raise Stop(message)


def check_command(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=10,env=ENV)
 require(p.returncode==0,'headroom command refused '+repr(cmd)+': '+p.stderr)
 return p.stdout


def members(pgid):
 raw=check_command(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,stat=,command='])
 result=[]
 for line in raw.splitlines():
  v=line.split(None,5)
  if len(v)==6 and int(v[2])==pgid:result.append({'pid':int(v[0]),'ppid':int(v[1]),'pgid':int(v[2]),'rss_kib':int(v[3]),'stat':v[4],'command':v[5]})
 # Linux containers may have a non-reaping PID1. This supervisor adopts only
 # its own orphaned descendants; never reap the retained leader before the
 # LAST group signal. macOS init already reaps the orphaned descendants.
 for row in result[:]:
  if row['pid']!=pgid and row['ppid']==os.getpid() and row['stat'].startswith('Z'):
   pid,status=os.waitpid(row['pid'],os.WNOHANG)
   if pid:ADOPTED_REAPS.append({'pid':pid,'wait_status':status,'owned_pgid':pgid});result.remove(row)
 return result


def publish_result(path,value):
 """Publish a complete once-only worker receipt, never a visible partial JSON."""
 path=Path(path);tmp=path.with_name(path.name+'.incoming')
 new_json(tmp,value)
 os.link(tmp,path)
 tmp.unlink()


def retain_group(argv,receipt):
 """Adapt spawn.py::retain_group: wait worker, publish status, retain PGID."""
 for sig in SIGNALS:signal.signal(sig,lambda *_:None)
 receipt=Path(receipt);worker=None
 try:
  worker=subprocess.Popen(argv,cwd=REPO,env=ENV)
  result={'pid':worker.pid,'exit_code':worker.wait(),'reaped':True,'error':None}
 except BaseException as e:
  result={'pid':worker.pid if worker else None,'exit_code':None,'reaped':False,'error':repr(e)}
  # If an unexpected post-launch error occurs, leave the worker in the owned
  # group. The parent still holds the leader and performs final group cleanup.
 try:publish_result(receipt/'worker-exit.json',result)
 except BaseException as e:
  print('WORKER_RESULT_WRITE_FAILURE: '+repr(e),file=sys.stderr,flush=True)
  return 125
 while True:time.sleep(1)


class OwnedGroup:
 """Only cleanup owns signals/reaping; a second call has no side effects."""
 def __init__(self,child):
  self.child=child;self.started=False
  self.result={'owned_pgid':child.pid,'signals':[],'errors':[],
               'leader_reaped':False,'leader_exit_code':None,'group_empty':False}
 def cleanup(self,receipt,reason):
  if self.started:return self.result
  self.started=True
  r=self.result;r['reason']=reason
  def error(where,e):
   value={'stage':where,'error':repr(e)}
   previous=next((x for x in r['errors'] if x['stage']==where and x['error']==value['error']),None)
   if previous is not None:previous['count']=previous.get('count',1)+1
   elif len(r['errors'])<64:r['errors'].append(value)
  def census():
   try:return members(self.child.pid)
   except BaseException as e:error('cleanup census',e);return None
  def record(name,value):
   try:new_json(Path(receipt)/name,value)
   except BaseException as e:error('write '+name,e)
  # This handle remains unreaped even when the helper has unexpectedly died.
  # All group signals precede the sole wait; no signal path exists after wait.
  for sig,grace in ((signal.SIGTERM,30),(signal.SIGKILL,10)):
   try:os.killpg(self.child.pid,sig);r['signals'].append(int(sig))
   except ProcessLookupError:r['signals'].append(int(sig))
   except BaseException as e:error('signal '+str(sig),e)
   deadline=time.monotonic()+grace
   while True:
    rows=census()
    if rows is not None and not [p for p in rows if p['pid']!=self.child.pid]:break
    if time.monotonic()>=deadline:break
    time.sleep(.1)
  # LAST group signal has now been attempted. Keep the lease on wait failure.
  warned=False
  while not r['leader_reaped']:
   try:
    r['leader_exit_code']=self.child.wait(timeout=10);r['leader_reaped']=True
   except BaseException as e:
    error('leader wait',e)
    if not warned:record('UNREAPED.json',r);warned=True
    time.sleep(2)
  # Census only after reap: never signal this numeric PGID again. Missing or
  # unreliable inspection cannot certify cleanup; retain lease for root recovery.
  deadline=time.monotonic()+10
  while True:
   rows=census()
   if rows==[]:r['group_empty']=True;break
   if time.monotonic()>=deadline and not warned:
    record('UNREAPED.json',r);warned=True
   time.sleep(.1 if not warned else 2)
  record('cleanup.json',r)
  return r


def cleanup(owned,receipt,reason):
 return owned.cleanup(receipt,reason)


def supervised(argv,receipt,timeout,cancel,monitor=False):
 receipt=Path(receipt);start=time.monotonic();owned=None;reason=None
 samples=[];initial=None;primary=None;worker=None;requested=[];old={};failures=[]
 def fail(where,e):failures.append({'stage':where,'error':repr(e)})
 def record(name,value):
  try:new_json(receipt/name,value)
  except BaseException as e:fail('write '+name,e)
 try:
  # Record, do not raise from a signal handler during Popen/receipt publication.
  # This also prevents repeat signals from interrupting cleanup or masking errors.
  for sig in SIGNALS:old[sig]=signal.signal(sig,lambda n,f:requested.append(n))
  with (receipt/'stdout.txt').open('x') as out,(receipt/'stderr.txt').open('x') as err:
   if monitor:initial=headroom();new_json(receipt/'headroom-before.json',initial)
   require(not cancel.exists(),'cancel requested before child')
   require(not requested,'supervisor signal before child')
   child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_group',json.dumps(argv),str(receipt)],cwd=REPO,env=ENV,stdout=out,stderr=err,start_new_session=True)
   owned=OwnedGroup(child)
   new_json(receipt/'child.json',{'pid':child.pid,'pgid':child.pid,'role':'retained group leader','worker_argv':argv,'utc':time.time(),'parent':os.getpid()})
   last=0
   while not (receipt/'worker-exit.json').exists():
    if requested:raise Stop('supervisor signal '+str(requested[0]))
    if cancel.exists():reason='cancel';raise Stop('child cancel')
    if time.monotonic()-start>timeout:reason='timeout';raise Stop('child timeout')
    ms=members(child.pid)
    leader=[p for p in ms if p['pid']==child.pid]
    require(len(leader)==1 and not leader[0]['stat'].startswith('Z'),'retained group leader exited before worker result')
    if monitor and time.monotonic()-last>=1:
     last=time.monotonic();h=headroom()
     samples.append({'utc':time.time(),'headroom':h,'members':ms})
     if h['swap_used_mib']>initial['swap_used_mib']+256:raise Stop('swap grew >256MiB')
     require(sum(p['rss_kib'] for p in ms)<=2*1024**2,'diagnostic owned-group RSS >2GiB')
    time.sleep(.05)
   worker=read(receipt/'worker-exit.json')
   require(worker.get('error') is None and worker.get('reaped') is True and type(worker.get('exit_code')) is int,'worker launch/wait failure '+str(worker.get('error')))
   if requested:raise Stop('supervisor signal '+str(requested[0]))
   if cancel.exists():reason='cancel';raise Stop('child cancel')
   if argv[0]=='/opt/homebrew/bin/timeout' and worker['exit_code'] in (124,137):
    reason='timeout';raise Stop('inner timeout expired')
   # The retained helper is expected; any other same-group process is an orphan.
   if [p for p in members(child.pid) if p['pid']!=child.pid]:
    reason='descendant-survived';raise Stop('owned descendants survived worker')
   if monitor:new_json(receipt/'headroom-after.json',headroom())
 except BaseException as e:
  primary=(e,e.__traceback__);reason=reason or str(e);fail('supervision',e)
 finally:
  try:
   cr=cleanup(owned,receipt,reason or 'normal exit') if owned else None
   if cr and cr['errors']:failures.extend(cr['errors'])
   if cr and cr['leader_exit_code']!=-signal.SIGKILL:fail('retained leader',Stop('retained leader unexpected exit '+str(cr['leader_exit_code'])))
   # Cleanup may have allowed the retained helper to reap a cancelled worker.
   if worker is None and (receipt/'worker-exit.json').exists():
    try:worker=read(receipt/'worker-exit.json')
    except BaseException as e:fail('read final worker result',e)
   if requested and not primary:fail('signal',Stop('supervisor signal '+str(requested[0])))
   record('monitor.json',samples)
   ex={'schema':'lloom-supervised-exit-v2','exit_code':worker.get('exit_code') if worker else None,
       'worker':worker,'cleanup':cr,'reaped':bool(cr and cr['leader_reaped'] and cr['group_empty']),
       'timed_out':reason=='timeout','cancelled':reason=='cancel',
       'signals_received':requested[:],'supervisor_error':failures[:] or None,
       'elapsed_seconds':time.monotonic()-start}
   record('exit.json',ex)
   if any(e['stage']=='write exit.json' for e in failures):
    # Disk/write failure can make an on-disk final receipt impossible. Stderr is
    # the fallback artifact; never return worker success without the receipt.
    print('FINAL_RECEIPT_WRITE_FAILURE: '+json.dumps({**ex,'supervisor_error':failures}),file=sys.stderr,flush=True)
  finally:
   for sig,handler in old.items():signal.signal(sig,handler)
 if primary:raise primary[0].with_traceback(primary[1])
 if failures:raise Stop('supervisor failure: '+repr(failures))
 return worker['exit_code']


def main():
 if sys.argv[1:2]==['_group']:
  return retain_group(json.loads(sys.argv[2]),sys.argv[3])
 receipt=Path(sys.argv[1]).resolve();receipt.mkdir()
 argv=sys.argv[2:];require(bool(argv),'missing child command')
 if sys.platform.startswith('linux'):
  libc=ctypes.CDLL(None,use_errno=True)
  require(libc.prctl(36,1,0,0,0)==0,'PR_SET_CHILD_SUBREAPER refused')
 try:
  return supervised(argv,receipt,10800,receipt/'CANCEL',monitor=False)
 except BaseException as error:
  print('COMMAND_GROUP_FAILURE: '+repr(error),file=sys.stderr,flush=True)
  if (receipt/'exit.json').exists():
   result=read(receipt/'exit.json')
   if result.get('signals_received'):return 128+result['signals_received'][0]
  return 125
 finally:
  new_json(receipt/'adopted-reaps.json',ADOPTED_REAPS)

if __name__=='__main__':sys.exit(main())
