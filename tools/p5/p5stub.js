// Headless p5 stub: drawing is a no-op, maths maps to Math.*, text() is captured.
global.__text = [];
const noop = () => {};
for (const f of ['createCanvas','background','noStroke','fill','stroke','square','circle',
                 'ellipse','rect','line','point','push','pop','translate','createButton',
                 'createSlider','loadPixels','updatePixels','strokeWeight','textSize',
                 'colorMode','frameRate','noLoop','loop','redraw','clear','beginShape',
                 'endShape','vertex','image','createGraphics','cursor','noCursor'])
  global[f] = noop;
global.text = (s,x,y) => { global.__text.push(String(s)); };
global.pow = Math.pow; global.sqrt = Math.sqrt; global.exp = Math.exp;
global.abs = Math.abs;  global.PI = Math.PI;   global.floor = Math.floor;
global.min = Math.min;  global.max = Math.max; global.round = Math.round;
global.log = Math.log;  global.sin = Math.sin; global.cos = Math.cos;
global.width = 400; global.height = 400;
