const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../index.html'), 'utf8');
function app() {
  let now = 10000;
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, { style: {}, textContent: '', classList: { toggle() {}, remove() {}, add() {} }, setAttribute() {}, getContext: () => ({clearRect() {}, setTransform() {}}), addEventListener() {}, matches: () => false });
    return nodes.get(id);
  };
  const context = vm.createContext({ console, Float32Array, Int32Array, Date: { now: () => now }, document: { getElementById: node, addEventListener() {} }, window: {addEventListener() {}}, ResizeObserver: class {observe() {}}, setTimeout, clearTimeout, cancelAnimationFrame() {} });
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  return { run: code => vm.runInContext(code, context), tick: ms => now += ms, node };
}
test('one letter is committed during a continuous hold', () => {
  const a = app(); let commits = 0;
  for (let i = 0; i < 100; i++) { commits += a.run("smoothPredict('A', .9, null).committed"); a.tick(50); }
  assert.equal(commits, 1);
});
test('low-confidence time cannot count toward a stable hold', () => {
  const a = app();
  for (let i=0;i<30;i++) { a.run("smoothPredict('A', .1, null)"); a.tick(50); }
  assert.equal(a.run("smoothPredict('A', .9, null).committed"), false);
});
test('current observation must agree with historical majority', () => {
  const a=app(); for(let i=0;i<20;i++) a.run("smoothPredict('A', .9, null)");
  a.tick(1000); assert.equal(a.run("smoothPredict('B', .9, null).committed"), false);
});
test('reset lets the same letter be signed again', () => {
  const a=app(); a.run("smoothPredict('A', .9, null)"); a.tick(800);
  assert.equal(a.run("smoothPredict('A', .9, null).committed"),true);
  a.run('resetRecognition()'); a.run("smoothPredict('A', .9, null)"); a.tick(800);
  assert.equal(a.run("smoothPredict('A', .9, null).committed"),true);
});
test('resampling matches training floor indices and zero padding', () => {
  const a=app();
  assert.equal(a.run('resampleBuffer(Array.from({length:45}, (_,i)=>new Float32Array(63).fill(i)))[63]'),1);
  assert.equal(a.run('resampleBuffer([new Float32Array(63).fill(9)])[63]'),0);
});
test('feature normalization is translation and scale invariant', () => {
  const a=app();
  assert.equal(a.run(`(() => { const lm=Array.from({length:21},(_,i)=>({x:i*.02,y:i*.03,z:i*.01})); const f=extractWordFeatures(lm); const g=extractWordFeatures(lm.map(p=>({x:p.x*2+3,y:p.y*2+4,z:p.z*2+1}))); return f.every((v,i)=>Math.abs(v-g[i])<1e-5); })()`),true);
});
test('word inference waits for a complete window',async()=>{
  const a=app(); a.run('wordModel = {}; wordBuffer = [new Float32Array(63)]');
  assert.equal(await a.run('inferWord()'),null);
});
test('word inference disposes every tensor after success',async()=>{
  const a=app(); a.run(`var disposed=0; var tensor={size:5,dispose(){disposed++}}; tf={tensor3d:()=>tensor,softmax:()=>({...tensor,data:async()=>[.8,.05,.05,.05,.05]}),dispose:t=>t.dispose()}; wordModel={executeAsync:async()=>tensor}; wordBuffer=Array.from({length:30},()=>new Float32Array(63));`);
  assert.equal((await a.run('inferWord()')).label,'hello'); assert.equal(a.run('disposed'),3);
});
test('word inference disposes input and output after failure',async()=>{
  const a=app(); a.run(`var disposed=0; var tensor={size:6,dispose(){disposed++}}; tf={tensor3d:()=>tensor,dispose:t=>t.dispose()}; wordModel={executeAsync:async()=>tensor}; wordBuffer=Array.from({length:30},()=>new Float32Array(63));`);
  await assert.rejects(a.run('inferWord()'),/Unexpected word model output/); assert.equal(a.run('disposed'),2);
});
test('one word request at a time and stale results are ignored',async()=>{
  const a=app(); a.run(`wordMode=true; state.cameraActive=true; var calls=0; var complete; inferWord=()=>{calls++;return new Promise(resolve=>complete=resolve)};`);
  const first=a.run('recognizeWord()'); await a.run('recognizeWord()'); assert.equal(a.run('calls'),1);
  a.run("resetRecognition(); complete({label:'hello',confidence:.9})"); await first;
  assert.equal(a.run('phraseTokens.length'),0); assert.equal(a.run('wordInFlight'),false);
});
test('phrase display replaces every underscore',()=>{
  const a=app(); a.run("addWordToPhrase('how_are_you')"); assert.equal(a.node('phraseBar').textContent,'how are you');
});
test('camera failure leaves a usable retry control',async()=>{
  const a=app(); a.run('navigator = {}'); await a.run('init()');
  assert.match(a.node('cameraMessage').textContent,/localhost or HTTPS/); assert.equal(a.node('startCamera').disabled,false);
});
test('camera startup requests one stream and stop releases it',async()=>{
  const a=app(); a.run(`var requests=0, stopped=0, sends=0;
    navigator={mediaDevices:{getUserMedia:async()=>{requests++;return {getTracks:()=>[{stop:()=>stopped++}],getVideoTracks:()=>[{addEventListener(){}}]}}}};
    Hands=class {setOptions(){} onResults(){} async send(){sends++}};
    requestAnimationFrame=()=>1;
    Object.assign(document.getElementById('video'),{play:async()=>{},videoWidth:960,videoHeight:540,readyState:2,currentTime:0});
    Object.assign(document.getElementById('videoContainer'),{clientWidth:960,clientHeight:540});`);
  await Promise.all([a.run('init()'),a.run('init()')]);
  assert.equal(a.run('requests'),1); assert.equal(a.run('state.cameraActive'),true);
  a.run('stopCamera()'); assert.equal(a.run('stopped'),1); assert.equal(a.run('state.cameraActive'),false);
});
test('model loading is shared and checks class order before loading weights',async()=>{
  const a=app(); a.run(`var loads=0, graphLoads=0; fetch=async()=>{loads++;return {ok:true,json:async()=>['incorrect']}}; tf={loadGraphModel:async()=>{graphLoads++;return {}}};`);
  await Promise.all([a.run('loadWordModel()'),a.run('loadWordModel()')]);
  assert.equal(a.run('loads'),1); assert.equal(a.run('graphLoads'),0); assert.equal(a.run('wordModel'),null);
  assert.match(a.node('modelStatus').textContent,/unavailable/);
});
