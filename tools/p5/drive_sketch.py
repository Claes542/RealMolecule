import sys, time, json
from cdp import Chrome
sid=sys.argv[1]; wait=int(sys.argv[2] if len(sys.argv)>2 else 600)
c=Chrome(); c.ws.settimeout(120)
c.goto("http://localhost:8001/run/%s.html?cb=%d"%(sid,int(wait))); time.sleep(4)
print("driving %s ..." % sid, flush=True)
t0=time.time(); last=None
while time.time()-t0 < wait:
    try: o=c.js("JSON.stringify(window.__read ? window.__read() : {})")
    except Exception as e: print("  (read failed: %s)"%e.__class__.__name__); time.sleep(5); continue
    if isinstance(o,str) and o.startswith('{'):
        d=json.loads(o)
        if d.get('t') != last:
            E=d.get('E', d.get('Etot'))
            print("   t=%-6s E=%-12s %s" % (d.get('t'),
                  (round(E,6) if isinstance(E,(int,float)) else E),
                  " ".join("%s=%s"%(k,round(v,4) if isinstance(v,float) else v)
                           for k,v in d.items() if k in ('Ekin','Epot','h','d'))), flush=True)
            last=d.get('t')
        if d.get('t') and d['t']>=3000: print("   DONE at t=%s"%d['t']); break
    time.sleep(20)
