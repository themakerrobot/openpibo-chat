// 블록 코딩(Blockly) 블록마다: 이름, 한국어 글, 툴팁, 고르는 값, 그 블록이 만드는 파이썬 코드를 뽑습니다.
// 기기 저장소(openpibo-os.pibo / .pibrain)의 ide/static/ 파일을 가짜 Blockly 위에서 그대로 돌려서 읽습니다.
//
//   node code_blocks.js <기기 저장소>/ide/static > blocks.json
//
// code_docs.py가 부릅니다.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const dir = process.argv[2];
if (!dir) { console.error("사용법: node code_blocks.js <ide/static 폴더>"); process.exit(2); }

const blocks = [];
const forBlock = {};
let defs = {};
const Python = {
  forBlock, ORDER_ATOMIC: 0, ORDER_NONE: 99, ORDER_FUNCTION_CALL: 2, ORDER_MEMBER: 1, ORDER_UNARY_SIGN: 4,
  ORDER_MULTIPLICATIVE: 5, ORDER_ADDITIVE: 6, ORDER_RELATIONAL: 11, ORDER_LOGICAL_NOT: 12, ORDER_LOGICAL_AND: 13,
  ORDER_LOGICAL_OR: 14, ORDER_CONDITIONAL: 15, ORDER_LAMBDA: 16, INDENT: "  ",
  get definitions_() { return defs; }, set definitions_(v) { defs = v; },
  valueToCode: (b, name) => `<${name}>`,
  statementToCode: (b, name) => `  <${name}>\n`,
  nameDB_: { getName: n => n }, variableDB_: { getName: n => n },
  quote_: s => JSON.stringify(String(s)),
  provideFunction_: (name) => name,
  injectId: (s) => s,
  addLoopTrap: (s) => s,
  prefixLines: (s, p) => s.split("\n").map(l => p + l).join("\n"),
};
const anything = new Proxy(function () {}, {
  get: (t, k) => k === Symbol.toPrimitive ? () => "" : anything,
  apply: () => anything, construct: () => anything,
});
const Blockly = new Proxy({
  Msg: {}, Python,
  defineBlocksWithJsonArray: arr => blocks.push(...arr),
  Blocks: {}, Extensions: { register: () => {}, registerMixin: () => {} },
  FieldDropdown: function () {}, Names: { NameType: { VARIABLE: "VARIABLE", PROCEDURE: "PROCEDURE" } },
}, { get: (t, k) => (k in t ? t[k] : anything) });

const ctx = vm.createContext({
  Blockly, console: { log() {}, warn() {}, error() {} }, window: anything, document: anything,
  navigator: anything, localStorage: anything, fetch: anything, setTimeout: () => 0, $: anything, jQuery: anything,
});
// 카테고리 이름: ko2en.js의 translations가 있으면 그 한국어 이름, 없으면 키 그대로 (device, motion …)
ctx.translations = new Proxy({}, { get: (t, k) => new Proxy({}, { get: () => String(k) }) });
for (const f of ["ko2en.js", "ko.js", "customblock.js", "customblock_callback.js", "customblock_toolbox.js"]) {
  const file = path.join(dir, f);
  if (!fs.existsSync(file)) {
    if (f === "ko2en.js") continue;
    console.error(file + " 없음"); process.exit(1);
  }
  try { vm.runInContext(fs.readFileSync(file, "utf8"), ctx, { filename: f }); }
  catch (e) { console.error(f + ": " + e.message); }
}

// 툴박스(학생 화면의 블록 상자)에 실제로 보이는 블록과 그 카테고리: 기기에 없는 블록은 정의만 남아 있음
const shown = {};
try {
  const tb = vm.runInContext("typeof toolbox === 'function' ? toolbox('ko') : null", ctx);
  (function walk(items, cat) {
    for (const it of items || []) {
      if (it.kind === "category") walk(it.contents, String(it.name || cat));
      else if (it.kind === "block" && it.type) shown[it.type] = cat;
    }
  })(tb && tb.contents, "");
} catch (e) { console.error("toolbox: " + e.message); }

function msg(s) {   // '%{BKY_X}' → 한국어 글
  return String(s || "").replace(/%\{BKY_([A-Z0-9_]+)\}/g, (m, k) => Blockly.Msg[k] !== undefined ? Blockly.Msg[k] : m);
}
function fakeBlock(def) {
  return {
    type: def.type, id: "b",
    getFieldValue: name => `<${name}>`,
    getInputTargetBlock: () => null, getInput: () => null, getField: () => null,
    getChildren: () => [], getNextBlock: () => null, getSurroundParent: () => null,
    itemCount_: 2, elseifCount_: 0, elseCount_: 0,
  };
}

const out = [];
for (const def of blocks) {
  if (!def || !def.type || !(def.type in shown)) continue;
  const args = (def.args0 || []).filter(a => a && a.type !== "field_image" && a.type !== "input_dummy");
  const fields = args.map(a => {
    const f = { name: a.name, type: a.type };
    if (a.check) f.check = a.check;
    if (Array.isArray(a.options)) f.options = a.options.map(o => Array.isArray(o) ? [msg(o[0]), o[1]] : o);
    if (a.value !== undefined) f.value = a.value;
    return f;
  });
  let code = null, imports = [];
  const fn = forBlock[def.type];
  if (typeof fn === "function") {
    defs = {};
    try {
      const r = fn.call(fakeBlock(def), fakeBlock(def));
      code = Array.isArray(r) ? r[0] : r;
      imports = Object.values(defs);
    } catch (e) { code = null; }
  }
  out.push({
    type: def.type, category: shown[def.type], text: msg(def.message0).replace(/%\d+/g, "□").replace(/\s+/g, " ").trim(),
    tooltip: msg(def.tooltip), fields, output: def.output !== undefined ? def.output : undefined,
    statement: !!(def.previousStatement || def.nextStatement), code, imports,
  });
}
process.stdout.write(JSON.stringify(out, null, 1));
