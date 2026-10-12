import sys, json, urllib.request
sid=sys.argv[1]
src=open('sk_%s.js'%sid).read()
# `let` at top level of a <script> is script-scoped, NOT on window, so append a reader
# INSIDE the same script block to close over the bindings.
html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>run %s</title></head><body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
<script>
%s
window.__read = function () {
  var o = {};
  try { o.E = E; } catch(e){}
  try { o.Etot = Etot; } catch(e){}
  try { o.Ekin = Ekin; } catch(e){}
  try { o.Epot = Epot; } catch(e){}
  try { o.E1 = E1; } catch(e){}
  try { o.E2 = E2; } catch(e){}
  try { o.E3 = E3; } catch(e){}
  try { o.t = t; } catch(e){}
  try { o.N = N; } catch(e){}
  try { o.M = M; } catch(e){}
  try { o.R1 = R1; } catch(e){}
  try { o.R = R; } catch(e){}
  try { o.h = h; } catch(e){}
  try { o.d = d; } catch(e){}
  try { o.D = D; } catch(e){}
  return o;
};
</script></body></html>""" % (sid, src)
open('run/%s.html'%sid,'w').write(html)
print("wrote run/%s.html (%d bytes)" % (sid, len(html)))
