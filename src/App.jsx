import React, { useState, useMemo, useRef, useCallback } from 'react';
import Papa from 'papaparse';
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import {
  Upload, Activity, AlertTriangle, CheckCircle2, ShieldAlert, SlidersHorizontal,
  Info, ChevronDown, ChevronUp, RotateCcw, Radio, Building2, FileWarning,
} from 'lucide-react';

/* ============================================================
   DESIGN TOKENS
   ============================================================ */
const C = {
  void: '#10141A',
  panel: '#1A2029',
  panelRaised: '#212836',
  line: '#2B3340',
  textPrimary: '#EDF0F3',
  textMuted: '#8B96A6',
  textFaint: '#5B6577',
  accent: '#5EEAD4',
  accentSoft: 'rgba(94,234,212,0.12)',
  io: '#5FD98A', ioSoft: 'rgba(95,217,138,0.13)',
  ls: '#F5B942', lsSoft: 'rgba(245,185,66,0.13)',
  cp: '#F2555F', cpSoft: 'rgba(242,85,95,0.13)',
};
const SANS = "'IBM Plex Sans', ui-sans-serif, system-ui, -apple-system, sans-serif";
const MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', 'JetBrains Mono', monospace";

/* ============================================================
   ENGINEERING / ML CORE
   Faithful to Muin, S. & Mosalam, K.M. (2021), "Structural Health
   Monitoring Using Machine Learning and Cumulative Absolute Velocity
   Features," Applied Sciences 11(12):5727. CAV = area under |a(t)|
   (Kramer, 1996). RCAV = CAV of the actual structural response over
   CAV of the corresponding LINEAR system under the same input.
   ============================================================ */

// CAV(t) = ∫|a(t)|dt  -- trapezoidal rule
function trapCAV(time, accel) {
  let cav = 0;
  for (let i = 0; i < time.length - 1; i++) {
    const dt = time[i + 1] - time[i];
    if (!(dt > 0)) continue;
    cav += 0.5 * (Math.abs(accel[i]) + Math.abs(accel[i + 1])) * dt;
  }
  return cav;
}

// Newmark-beta, constant average acceleration (gamma=1/2, beta=1/4).
// Linear SDOF (mass normalized to 1) under base excitation ag(t):
//   u'' + 2*zeta*omega*u' + omega^2*u = -ag(t)
// Returns the TOTAL (absolute) acceleration response = u'' + ag,
// i.e. what an accelerometer at the mass (roof) would read if the
// structure stayed perfectly elastic.
function newmarkLinearSDOF(time, ag, T1, zeta) {
  const n = ag.length;
  const omega = (2 * Math.PI) / Math.max(T1, 1e-3);
  const k = omega * omega;
  const c = 2 * Math.max(zeta, 0) * omega;
  const gamma = 0.5, beta = 0.25;
  const u = new Float64Array(n), v = new Float64Array(n), a = new Float64Array(n);
  a[0] = -ag[0] - c * v[0] - k * u[0];
  const aTotal = new Float64Array(n);
  aTotal[0] = a[0] + ag[0];
  for (let i = 0; i < n - 1; i++) {
    const dt = time[i + 1] - time[i];
    if (!(dt > 0)) { aTotal[i + 1] = aTotal[i]; continue; }
    const dp = -(ag[i + 1] - ag[i]);
    const kHat = k + 1 / (beta * dt * dt) + (gamma * c) / (beta * dt);
    const A = 1 / (beta * dt) + (gamma / beta) * c;
    const B = 1 / (2 * beta) + dt * (gamma / (2 * beta) - 1) * c;
    const dPHat = dp + A * v[i] + B * a[i];
    const du = dPHat / kHat;
    const dv = (gamma / (beta * dt)) * du - (gamma / beta) * v[i] + dt * (1 - gamma / (2 * beta)) * a[i];
    const da = (1 / (beta * dt * dt)) * du - (1 / (beta * dt)) * v[i] - (1 / (2 * beta)) * a[i];
    u[i + 1] = u[i] + du; v[i + 1] = v[i] + dv; a[i + 1] = a[i] + da;
    aTotal[i + 1] = a[i + 1] + ag[i + 1];
  }
  return Array.from(aTotal);
}

// Deterministic synthetic accelerogram for the in-app "sample data" demo.
// NOT a recorded or spectrum-matched ground motion -- illustrative only.
function generateSyntheticGround(duration = 20, dt = 0.02, pgaTarget = 0.06) {
  const n = Math.round(duration / dt) + 1;
  const time = new Array(n), raw = new Array(n);
  const freqs = [0.8, 1.5, 2.3, 3.4, 5.0, 7.2, 10.5, 14.0];
  const phases = [0.3, 2.1, 4.4, 1.0, 3.3, 0.7, 5.1, 2.8];
  const amps = freqs.map((f) => 1 / Math.pow(f, 0.9));
  const t1 = 1.5, t2 = 5.5, decayC = 0.35;
  let peak = 0;
  for (let i = 0; i < n; i++) {
    const t = i * dt; time[i] = t;
    let env;
    if (t < t1) env = Math.pow(t / t1, 2);
    else if (t < t2) env = 1.0;
    else env = Math.exp(-decayC * (t - t2));
    let s = 0;
    for (let k = 0; k < freqs.length; k++) s += amps[k] * Math.sin(2 * Math.PI * freqs[k] * t + phases[k]);
    raw[i] = env * s;
    if (Math.abs(raw[i]) > peak) peak = Math.abs(raw[i]);
  }
  const scale = pgaTarget / peak;
  return { time, ground: raw.map((v) => v * scale) };
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// Ordinal Logistic Regression (proportional-odds / cumulative-logit model),
// same structure as Muin & Mosalam (2021) Eqs. 9-14, reduced to 3 ordered
// classes (IO < LS < CP) per this prototype's brief.
// ILLUSTRATIVE parameters -- see methodology panel. NOT fitted from
// labeled damage data.
const OLR = { betaCAV: 8.0, betaD: 12.0, thetaIOLS: 3.08, thetaLSCP: 5.04 };
function classifyOLR(cav, rcav) {
  const D = Math.max(0, 1 - rcav); // "softening index": 0 = purely elastic
  const eta = OLR.betaCAV * cav + OLR.betaD * D;
  const pIOcum = sigmoid(OLR.thetaIOLS - eta);
  const pLScum = sigmoid(OLR.thetaLSCP - eta);
  const pIO = pIOcum, pLS = pLScum - pIOcum, pCP = 1 - pLScum;
  const entries = [['IO', pIO], ['LS', pLS], ['CP', pCP]];
  entries.sort((a, b) => b[1] - a[1]);
  return { pIO, pLS, pCP, predicted: entries[0][0], D, eta };
}

function downsample(arr, maxPoints) {
  if (arr.length <= maxPoints) return arr;
  const step = Math.ceil(arr.length / maxPoints);
  return arr.filter((_, i) => i % step === 0);
}

function findColumn(fields, aliases) {
  const norm = (f) => String(f).trim().toLowerCase().replace(/[^a-z0-9]/g, '');
  const normFields = fields.map(norm);
  for (const alias of aliases) {
    const idx = normFields.indexOf(norm(alias));
    if (idx !== -1) return fields[idx];
  }
  return null;
}

const G0 = 980.665; // cm/s^2 per g

/* ============================================================
   STATIC CONTENT
   ============================================================ */
const REC = {
  IO: {
    title: 'Immediate Occupancy', code: 'IO', color: C.io, soft: C.ioSoft,
    asce: 'ASCE 41-17 §2.3.1.1 (S-1)',
    summary: 'Kerusakan struktural diperkirakan minimal (retak rambut, jika ada). Elemen struktur diperkirakan tetap mempertahankan kekuatan dan kekakuannya.',
    actions: [
      'Bangunan kemungkinan besar dapat dihuni kembali segera.',
      'Tetap lakukan verifikasi visual cepat oleh petugas/insinyur sebelum penghunian penuh.',
      'Prioritaskan verifikasi untuk fasilitas kritis (RS, sekolah, kantor tanggap darurat).',
    ],
  },
  LS: {
    title: 'Life Safety', code: 'LS', color: C.ls, soft: C.lsSoft,
    asce: 'ASCE 41-17 §2.3.1.3 (S-3)',
    summary: 'Indikasi kerusakan struktural sedang. Bangunan diperkirakan tidak runtuh, namun kemungkinan tidak ekonomis diperbaiki tanpa evaluasi lanjutan.',
    actions: [
      'Batasi akses/penghunian sampai ada inspeksi lanjutan.',
      'Jadwalkan inspeksi visual + instrumentasi oleh insinyur struktur berlisensi.',
      'Jangan gunakan secara normal sebelum status diperbarui.',
    ],
  },
  CP: {
    title: 'Collapse Prevention', code: 'CP', color: C.cp, soft: C.cpSoft,
    asce: 'ASCE 41-17 §2.3.1.5 (S-5)',
    summary: 'Indikasi kerusakan struktural berat, mendekati ambang keruntuhan. Elemen vertikal mungkin masih menahan beban gravitasi, namun kapasitas lateral tersisa rendah.',
    actions: [
      'JANGAN memasuki bangunan. Evakuasi dan pasang batas akses segera.',
      'Laporkan ke otoritas kebencanaan (BPBD) untuk tagging darurat.',
      'Perlu evaluasi detail (Tier 2/3 ASCE 41 atau SNI 9274:2025) sebelum keputusan perbaikan/rehabilitasi/pembongkaran.',
    ],
  },
};

/* ============================================================
   SMALL UI PIECES
   ============================================================ */
function Eyebrow({ children }) {
  return (
    <div style={{ color: C.accent, fontFamily: MONO, letterSpacing: '0.14em' }} className="text-xs uppercase font-medium">
      {children}
    </div>
  );
}

function Panel({ children, className = '' }) {
  return (
    <div
      style={{ background: C.panel, border: `1px solid ${C.line}`, borderRadius: 14 }}
      className={`p-4 sm:p-5 ${className}`}
    >
      {children}
    </div>
  );
}

function ModeButton({ active, onClick, icon: Icon, label, sub }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? C.accentSoft : 'transparent',
        border: `1px solid ${active ? C.accent : C.line}`,
        borderRadius: 12,
        color: active ? C.textPrimary : C.textMuted,
      }}
      className="flex-1 flex items-center gap-3 px-4 py-3 text-left transition-colors"
    >
      <Icon size={18} style={{ color: active ? C.accent : C.textFaint, flexShrink: 0 }} />
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div style={{ color: C.textFaint }} className="text-xs">{sub}</div>
      </div>
    </button>
  );
}

function StatCard({ label, value, unit, caption, badge }) {
  return (
    <div style={{ background: C.panelRaised, border: `1px solid ${C.line}`, borderRadius: 12 }} className="p-4">
      <div style={{ color: C.textMuted }} className="text-xs uppercase tracking-wide font-medium mb-1">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <span style={{ color: C.textPrimary, fontFamily: MONO }} className="text-2xl sm:text-3xl font-semibold tabular-nums">{value}</span>
        {unit && <span style={{ color: C.textMuted, fontFamily: MONO }} className="text-sm">{unit}</span>}
      </div>
      {caption && <div style={{ color: C.textFaint }} className="text-xs mt-1.5 leading-snug">{caption}</div>}
      {badge && (
        <div style={{ color: C.accent, background: C.accentSoft, borderRadius: 999 }} className="inline-block text-xs px-2 py-0.5 mt-2">
          {badge}
        </div>
      )}
    </div>
  );
}

function ProbGauge({ code, title, value, color, soft, active }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div style={{ background: active ? soft : 'transparent', border: `1px solid ${active ? color : C.line}`, borderRadius: 10 }} className="p-3 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div>
          <span style={{ color: C.textFaint, fontFamily: MONO }} className="text-xs tracking-widest uppercase">{code}</span>
          <div style={{ color: C.textPrimary }} className="text-sm font-medium">{title}</div>
        </div>
        <div style={{ color, fontFamily: MONO }} className="text-xl font-semibold tabular-nums">{pct.toFixed(1)}%</div>
      </div>
      <div style={{ background: C.line, borderRadius: 999 }} className="h-2 overflow-hidden">
        <div style={{ width: `${pct}%`, background: color, borderRadius: 999, transition: 'width 500ms ease' }} className="h-full" />
      </div>
    </div>
  );
}

function Field({ label, children, hint }) {
  return (
    <div>
      <label style={{ color: C.textMuted }} className="text-xs uppercase tracking-wide font-medium block mb-1.5">{label}</label>
      {children}
      {hint && <div style={{ color: C.textFaint }} className="text-xs mt-1">{hint}</div>}
    </div>
  );
}

const inputStyle = {
  background: C.panelRaised, border: `1px solid ${C.line}`, borderRadius: 8,
  color: C.textPrimary, fontFamily: MONO,
};

/* ============================================================
   MAIN APP
   ============================================================ */
export default function App() {
  const [mode, setMode] = useState('manual');
  const [manualCAV, setManualCAV] = useState(0.12);
  const [manualRCAV, setManualRCAV] = useState(0.95);

  const [csvData, setCsvData] = useState(null); // {time, ground, roof, name}
  const [csvError, setCsvError] = useState(null);
  const [csvWarning, setCsvWarning] = useState(null);
  const [T1, setT1] = useState(0.45);
  const [zeta, setZeta] = useState(0.05);
  const [unit, setUnit] = useState('g');
  const [methodOpen, setMethodOpen] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setCsvError(null); setCsvWarning(null);
    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = String(evt.target.result);
      Papa.parse(text, {
        header: true, dynamicTyping: true, skipEmptyLines: true,
        complete: (results) => {
          const fields = results.meta.fields || [];
          let timeCol = findColumn(fields, ['time_s', 'time', 't', 'waktu', 'detik']);
          let groundCol = findColumn(fields, ['acc_ground_g', 'acc_ground', 'ground', 'dasar', 'base']);
          let roofCol = findColumn(fields, ['acc_roof_g', 'acc_roof', 'roof', 'atap', 'top']);
          let usedFallback = false;
          if ((!timeCol || !groundCol || !roofCol) && fields.length >= 3) {
            timeCol = timeCol || fields[0];
            groundCol = groundCol || fields[1];
            roofCol = roofCol || fields[2];
            usedFallback = true;
          }
          if (!timeCol || !groundCol || !roofCol) {
            setCsvError('CSV harus memiliki minimal 3 kolom: waktu, percepatan dasar, dan percepatan atap.');
            return;
          }
          const rows = results.data.filter((r) => r[timeCol] != null && r[groundCol] != null && r[roofCol] != null);
          if (rows.length < 10) {
            setCsvError('Data valid terlalu sedikit (<10 baris) setelah parsing. Periksa format CSV.');
            return;
          }
          const time = rows.map((r) => Number(r[timeCol]));
          let ground = rows.map((r) => Number(r[groundCol]));
          let roof = rows.map((r) => Number(r[roofCol]));
          if (unit === 'cms2') { ground = ground.map((v) => v / G0); roof = roof.map((v) => v / G0); }
          if (usedFallback) setCsvWarning(`Header kolom tidak dikenali — menggunakan urutan kolom (1=waktu, 2=dasar, 3=atap). Ditemukan: "${timeCol}", "${groundCol}", "${roofCol}".`);
          setCsvData({ time, ground, roof, name: file.name });
        },
        error: () => setCsvError('Gagal membaca file CSV. Pastikan formatnya valid (koma sebagai pemisah).'),
      });
    };
    reader.readAsText(file);
  }, [unit]);

  const loadSample = useCallback(() => {
    setCsvError(null); setCsvWarning(null);
    const { time, ground } = generateSyntheticGround(20, 0.02, 0.06);
    const linear = newmarkLinearSDOF(time, ground, T1, zeta);
    const softening = 0.95;
    const roof = linear.map((v, i) => v * softening + 0.01 * Math.sin(37 * time[i]) * 0.02);
    setUnit('g');
    setCsvData({ time, ground, roof, name: 'Acceleration_sample.csv (sintetis)' });
  }, [T1, zeta]);

  const resetCsv = useCallback(() => {
    setCsvData(null); setCsvError(null); setCsvWarning(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  const csvFeatures = useMemo(() => {
    if (!csvData) return null;
    const { time, ground, roof } = csvData;
    const cavGround = trapCAV(time, ground);
    const linearRoof = newmarkLinearSDOF(time, ground, T1, zeta);
    const cavLinear = trapCAV(time, linearRoof);
    const cavRoofMeasured = trapCAV(time, roof);
    const rcav = cavLinear > 1e-9 ? cavRoofMeasured / cavLinear : null;

    const dtVals = [];
    for (let i = 0; i < time.length - 1; i++) dtVals.push(time[i + 1] - time[i]);
    const dtAvg = dtVals.reduce((a, b) => a + b, 0) / Math.max(dtVals.length, 1);
    const coarseWarning = dtAvg > T1 / 10;

    const idx = downsample(time.map((_, i) => i), 400);
    const chartData = idx.map((i) => ({
      t: Number(time[i].toFixed(3)),
      ground: ground[i],
      roof: roof[i],
      absGround: Math.abs(ground[i]),
    }));

    return { cavGround, cavLinear, cavRoofMeasured, rcav, dtAvg, coarseWarning, chartData };
  }, [csvData, T1, zeta]);

  const activeCAV = mode === 'manual' ? manualCAV : (csvFeatures ? csvFeatures.cavGround : null);
  const activeRCAV = mode === 'manual' ? manualRCAV : (csvFeatures ? csvFeatures.rcav : null);
  const classification = useMemo(
    () => (activeCAV != null && activeRCAV != null ? classifyOLR(activeCAV, activeRCAV) : null),
    [activeCAV, activeRCAV]
  );
  const rec = classification ? REC[classification.predicted] : null;

  return (
    <div style={{ background: C.void, fontFamily: SANS, color: C.textPrimary, minHeight: '100%' }} className="w-full">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        input[type="range"] { -webkit-appearance: none; appearance: none; height: 4px; border-radius: 999px; background: ${C.line}; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: ${C.accent}; cursor: pointer; border: 2px solid ${C.void}; box-shadow: 0 0 0 1px ${C.accent}; }
        input[type="range"]::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: ${C.accent}; cursor: pointer; border: 2px solid ${C.void}; }
        .no-spin::-webkit-outer-spin-button, .no-spin::-webkit-inner-spin-button { -webkit-appearance: none; margin:0; }
      `}</style>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 sm:py-8">

        {/* MASTHEAD */}
        <header style={{ position: 'relative', overflow: 'hidden', borderBottom: `1px solid ${C.line}` }} className="pb-6 mb-6">
          <svg viewBox="0 0 600 60" preserveAspectRatio="none" style={{ position: 'absolute', top: 6, left: 0, width: '100%', height: 48, opacity: 0.14 }}>
            <polyline
              fill="none" stroke={C.accent} strokeWidth="1.5"
              points="0,30 20,30 35,10 50,48 65,18 80,36 95,30 115,30 130,4 145,52 160,14 175,40 195,30 600,30"
            />
          </svg>
          <div style={{ position: 'relative' }}>
            <Eyebrow>PROTOTIPE · STRUCTURAL HEALTH MONITORING</Eyebrow>
            <h1 style={{ color: C.textPrimary }} className="text-xl sm:text-2xl font-semibold mt-1 leading-tight">
              AI-Based Rapid Post-Earthquake Building Assessment
            </h1>
            <p style={{ color: C.textMuted }} className="text-sm mt-1.5 max-w-2xl">
              Klasifikasi kondisi bangunan pasca-gempa dari fitur CAV / RCAV menggunakan Ordinal Logistic Regression.
            </p>
            <div style={{ color: C.ls, background: C.lsSoft, borderRadius: 8 }} className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 mt-3">
              <AlertTriangle size={13} />
              Alat bantu skrining awal — bukan pengganti inspeksi insinyur berlisensi
            </div>
          </div>
        </header>

        {/* MODE TOGGLE */}
        <div className="flex flex-col sm:flex-row gap-2 mb-5">
          <ModeButton active={mode === 'manual'} onClick={() => setMode('manual')} icon={SlidersHorizontal} label="Simulasi Manual" sub="Atur CAV & RCAV langsung" />
          <ModeButton active={mode === 'csv'} onClick={() => setMode('csv')} icon={Upload} label="Unggah Data Akselerasi" sub="Acceleration.csv → pipeline penuh" />
        </div>

        {/* INPUT PANEL */}
        {mode === 'manual' ? (
          <Panel className="mb-5">
            <div className="grid sm:grid-cols-2 gap-5">
              <div>
                <div className="flex items-baseline justify-between mb-1">
                  <label style={{ color: C.textMuted }} className="text-xs uppercase tracking-wide font-medium">CAV — Cumulative Absolute Velocity</label>
                  <span style={{ color: C.accent, fontFamily: MONO }} className="text-sm font-medium tabular-nums">{manualCAV.toFixed(3)} g·det</span>
                </div>
                <input type="range" min={0} max={0.5} step={0.005} value={manualCAV} onChange={(e) => setManualCAV(Number(e.target.value))} className="w-full" />
                <div style={{ color: C.textFaint }} className="text-xs mt-1.5 leading-snug">
                  Referensi: ambang klasik NRC/EPRI ≈ 0.16 g·detik untuk onset kerusakan bangunan berdesain baik (Whittier, 1987) — konteks kriteria shutdown PLTN, ditampilkan hanya sebagai perbandingan skala.
                </div>
              </div>
              <div>
                <div className="flex items-baseline justify-between mb-1">
                  <label style={{ color: C.textMuted }} className="text-xs uppercase tracking-wide font-medium">RCAV — Relative CAV</label>
                  <span style={{ color: C.accent, fontFamily: MONO }} className="text-sm font-medium tabular-nums">{manualRCAV.toFixed(3)}</span>
                </div>
                <input type="range" min={0.4} max={1.1} step={0.01} value={manualRCAV} onChange={(e) => setManualRCAV(Number(e.target.value))} className="w-full" />
                <div style={{ color: C.textFaint }} className="text-xs mt-1.5 leading-snug">
                  1.0 = respons struktur sama seperti model linear (tanpa pelunakan). Menurun seiring bertambahnya kerusakan/perpanjangan periode.
                </div>
              </div>
            </div>
          </Panel>
        ) : (
          <Panel className="mb-5">
            <div className="grid sm:grid-cols-3 gap-4 mb-4">
              <Field label="Periode getar fundamental, T1">
                <input type="number" step="0.05" min="0.1" max="4" value={T1} onChange={(e) => setT1(Math.max(0.1, Number(e.target.value) || 0.1))}
                  style={inputStyle} className="no-spin w-full px-3 py-2 text-sm" />
              </Field>
              <Field label="Rasio redaman, ζ">
                <input type="number" step="0.01" min="0.01" max="0.2" value={zeta} onChange={(e) => setZeta(Math.max(0.01, Number(e.target.value) || 0.01))}
                  style={inputStyle} className="no-spin w-full px-3 py-2 text-sm" />
              </Field>
              <Field label="Satuan percepatan pada CSV">
                <select value={unit} onChange={(e) => setUnit(e.target.value)} style={inputStyle} className="w-full px-3 py-2 text-sm">
                  <option value="g">g</option>
                  <option value="cms2">cm/s² (gal)</option>
                </select>
              </Field>
            </div>
            <div style={{ color: C.textFaint }} className="text-xs mb-4 leading-snug">
              T1 &amp; ζ mendefinisikan model SDOF linear referensi (integrasi Newmark-β) yang dipakai untuk menghitung RCAV. Idealnya diambil dari identifikasi sistem / uji getar ambien bangunan aktual, bukan ditebak.
            </div>

            <div className="flex flex-wrap items-center gap-2 mb-3">
              <label style={{ background: C.panelRaised, border: `1px solid ${C.line}`, borderRadius: 8, color: C.textPrimary }} className="cursor-pointer text-sm px-3.5 py-2 flex items-center gap-2">
                <Upload size={15} style={{ color: C.accent }} />
                Unggah Acceleration.csv
                <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFile} style={{ display: 'none' }} />
              </label>
              <button onClick={loadSample} style={{ background: 'transparent', border: `1px solid ${C.line}`, borderRadius: 8, color: C.textMuted }} className="text-sm px-3.5 py-2 flex items-center gap-2">
                <Radio size={15} />
                Muat Data Contoh
              </button>
              {csvData && (
                <button onClick={resetCsv} style={{ background: 'transparent', border: `1px solid ${C.line}`, borderRadius: 8, color: C.textFaint }} className="text-sm px-3.5 py-2 flex items-center gap-2">
                  <RotateCcw size={14} /> Reset
                </button>
              )}
            </div>

            <div style={{ color: C.textFaint }} className="text-xs leading-snug mb-1">
              Format kolom yang dikenali: <span style={{ fontFamily: MONO, color: C.textMuted }}>time_s, acc_ground_g, acc_roof_g</span> (header lain dicoba dikenali otomatis; jika gagal, kolom 1/2/3 dipakai sebagai fallback).
            </div>

            {csvError && (
              <div style={{ color: C.cp, background: C.cpSoft, borderRadius: 8 }} className="text-sm px-3 py-2 mt-2 flex items-start gap-2">
                <FileWarning size={15} style={{ flexShrink: 0, marginTop: 2 }} /> {csvError}
              </div>
            )}
            {csvWarning && (
              <div style={{ color: C.ls, background: C.lsSoft, borderRadius: 8 }} className="text-sm px-3 py-2 mt-2 flex items-start gap-2">
                <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 2 }} /> {csvWarning}
              </div>
            )}
            {csvData && (
              <div style={{ color: C.textFaint }} className="text-xs mt-2">
                Sumber: <span style={{ color: C.textMuted, fontFamily: MONO }}>{csvData.name}</span> · {csvData.time.length} baris
                {csvFeatures && csvFeatures.coarseWarning && (
                  <span style={{ color: C.ls }}> · Δt rata-rata ({csvFeatures.dtAvg.toFixed(4)}s) relatif kasar terhadap T1 — akurasi integrasi Newmark-β dapat berkurang.</span>
                )}
              </div>
            )}
          </Panel>
        )}

        {/* WAVEFORM (CSV mode only, once data present) */}
        {mode === 'csv' && csvFeatures && (
          <Panel className="mb-5">
            <div className="flex items-center gap-2 mb-3">
              <Activity size={15} style={{ color: C.accent }} />
              <span style={{ color: C.textPrimary }} className="text-sm font-medium">Riwayat Percepatan</span>
              <span style={{ color: C.textFaint }} className="text-xs">— area terarsir = |percepatan dasar(t)|, integralnya adalah CAV</span>
            </div>
            <div style={{ width: '100%', height: 200 }}>
              <ResponsiveContainer>
                <ComposedChart data={csvFeatures.chartData} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke={C.line} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" stroke={C.textFaint} tick={{ fontSize: 11, fontFamily: MONO, fill: C.textFaint }} label={{ value: 'detik', position: 'insideBottomRight', offset: -2, fill: C.textFaint, fontSize: 11 }} />
                  <YAxis stroke={C.textFaint} tick={{ fontSize: 11, fontFamily: MONO, fill: C.textFaint }} />
                  <Tooltip
                    contentStyle={{ background: C.panelRaised, border: `1px solid ${C.line}`, borderRadius: 8, fontFamily: MONO, fontSize: 12 }}
                    labelStyle={{ color: C.textMuted }} itemStyle={{ color: C.textPrimary }}
                  />
                  <Area type="monotone" dataKey="absGround" stroke="none" fill={C.accent} fillOpacity={0.12} name="|dasar(t)|" />
                  <Line type="monotone" dataKey="ground" stroke={C.accent} dot={false} strokeWidth={1.4} name="dasar" />
                  <Line type="monotone" dataKey="roof" stroke={C.ls} dot={false} strokeWidth={1.2} strokeDasharray="4 3" name="atap" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        )}

        {/* FEATURE CARDS */}
        {(mode === 'manual' || csvFeatures) && (
          <div className="grid sm:grid-cols-2 gap-4 mb-5">
            <StatCard
              label="CAV (lantai dasar)"
              value={activeCAV != null ? activeCAV.toFixed(3) : '—'}
              unit="g·det"
              caption="∫|percepatan dasar(t)| dt — Muin & Mosalam (2021), Eq. 2"
              badge={activeCAV != null && activeCAV >= 0.16 ? '≥ ambang referensi NRC/EPRI' : null}
            />
            <StatCard
              label="RCAV (atap)"
              value={activeRCAV != null ? activeRCAV.toFixed(3) : '—'}
              unit=""
              caption={mode === 'csv' && csvFeatures
                ? `CAVₛ (${csvFeatures.cavRoofMeasured.toFixed(3)}) / CAVₗ (${csvFeatures.cavLinear.toFixed(3)}) — respons aktual atap dibagi respons SDOF linear`
                : 'CAVₛ / CAVₗ — respons aktual struktur dibagi respons model linear pada eksitasi yang sama'}
            />
          </div>
        )}

        {/* CLASSIFICATION */}
        {classification && (
          <Panel className="mb-5">
            <div className="flex items-center gap-2 mb-3">
              <span style={{ color: C.textPrimary }} className="text-sm font-medium">Klasifikasi — Ordinal Logistic Regression</span>
            </div>
            <div style={{ background: C.panelRaised, border: `1px solid ${C.line}`, borderRadius: 8, color: C.textMuted, fontFamily: MONO }} className="text-xs px-3 py-2.5 mb-4 leading-relaxed overflow-x-auto whitespace-nowrap">
              η = {OLR.betaCAV.toFixed(1)}·CAV + {OLR.betaD.toFixed(1)}·D,&nbsp; D = max(0, 1−RCAV) = {classification.D.toFixed(3)},&nbsp; η = {classification.eta.toFixed(3)}
              <br />
              P(Y≤IO) = σ({OLR.thetaIOLS.toFixed(2)} − η) &nbsp;|&nbsp; P(Y≤LS) = σ({OLR.thetaLSCP.toFixed(2)} − η) &nbsp;|&nbsp; σ(x) = 1/(1+e⁻ˣ)
            </div>
            <div className="grid gap-2.5">
              <ProbGauge code="IO" title="Immediate Occupancy" value={classification.pIO} color={C.io} soft={C.ioSoft} active={classification.predicted === 'IO'} />
              <ProbGauge code="LS" title="Life Safety" value={classification.pLS} color={C.ls} soft={C.lsSoft} active={classification.predicted === 'LS'} />
              <ProbGauge code="CP" title="Collapse Prevention" value={classification.pCP} color={C.cp} soft={C.cpSoft} active={classification.predicted === 'CP'} />
            </div>
          </Panel>
        )}

        {/* RECOMMENDATION */}
        {rec && (
          <Panel className="mb-5">
            <div className="flex items-start gap-3">
              <div style={{ background: rec.soft, borderRadius: 10, padding: 8, flexShrink: 0 }}>
                {rec.code === 'IO' && <CheckCircle2 size={20} style={{ color: rec.color }} />}
                {rec.code === 'LS' && <AlertTriangle size={20} style={{ color: rec.color }} />}
                {rec.code === 'CP' && <ShieldAlert size={20} style={{ color: rec.color }} />}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span style={{ color: rec.color }} className="text-base font-semibold">{rec.title}</span>
                  <span style={{ color: C.textFaint, fontFamily: MONO }} className="text-xs">{rec.asce}</span>
                </div>
                <p style={{ color: C.textMuted }} className="text-sm mt-1.5 leading-relaxed">{rec.summary}</p>
                <ul className="mt-3 space-y-1.5">
                  {rec.actions.map((a, i) => (
                    <li key={i} style={{ color: C.textPrimary }} className="text-sm flex items-start gap-2">
                      <span style={{ color: rec.color }} className="mt-1.5" >●</span>
                      <span>{a}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </Panel>
        )}

        {/* METHODOLOGY / LIMITATIONS */}
        <Panel className="mb-6">
          <button onClick={() => setMethodOpen((v) => !v)} className="w-full flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Info size={15} style={{ color: C.accent }} />
              <span style={{ color: C.textPrimary }} className="text-sm font-medium">Metodologi &amp; Keterbatasan Model</span>
            </span>
            {methodOpen ? <ChevronUp size={16} style={{ color: C.textMuted }} /> : <ChevronDown size={16} style={{ color: C.textMuted }} />}
          </button>
          {methodOpen && (
            <div style={{ color: C.textMuted }} className="text-sm mt-4 space-y-4 leading-relaxed">
              <div>
                <div style={{ color: C.textPrimary }} className="font-medium mb-1">Basis metodologi</div>
                <p>Diadaptasi dari Muin, S. &amp; Mosalam, K.M. (2021), "Structural Health Monitoring Using Machine Learning and Cumulative Absolute Velocity Features," <i>Applied Sciences</i>, 11(12):5727 — dievaluasi pada model SDOF/MDOF numerik dan bangunan terinstrumentasi (Tai-Tung Fire Bureau Building, Taiwan).</p>
              </div>
              <div>
                <div style={{ color: C.textPrimary }} className="font-medium mb-1">Rumus fitur</div>
                <div style={{ fontFamily: MONO, background: C.panelRaised, borderRadius: 8 }} className="text-xs p-3 space-y-1">
                  <div>CAV(t) = ∫₀ᵀ |percepatan(t)| dt &nbsp;(Kramer, 1996)</div>
                  <div>RCAV = CAVₛ / CAVₗ &nbsp;— CAVₛ: respons aktual; CAVₗ: respons SDOF linear (Newmark-β, T1, ζ) pada input sama</div>
                </div>
              </div>
              <div>
                <div style={{ color: C.textPrimary }} className="font-medium mb-1">Perbedaan penting dari paper asli</div>
                <p>Paper sumber menggunakan 4 kelas kerusakan berbasis daktilitas perpindahan (FEMA P-58: tidak rusak / ringan / sedang / berat), <b>bukan</b> label ASCE 41 IO/LS/CP secara langsung. Prototipe ini memetakan ulang ke IO/LS/CP (S-1/S-3/S-5) agar keluarannya langsung dapat ditindaklanjuti secara operasional — pemetaan ini adalah lapisan tambahan yang perlu divalidasi terhadap batas parameter kinerja ASCE 41-17, bukan hasil langsung dari paper.</p>
              </div>
              <div>
                <div style={{ color: C.textPrimary }} className="font-medium mb-1">Keterbatasan prototipe ini</div>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Koefisien OLR (β_CAV={OLR.betaCAV}, β_D={OLR.betaD}, θ_IO/LS={OLR.thetaIOLS}, θ_LS/CP={OLR.thetaLSCP}) bersifat <b>ilustratif</b> — dikalibrasi agar konsisten secara kualitatif, <b>bukan</b> hasil regresi pada data kerusakan riil.</li>
                  <li>CAVₗ dihitung dari SDOF satu derajat kebebasan; bangunan nyata bersifat MDOF dengan banyak ragam getar — pendekatan ini menyederhanakan sebagaimana paper asli lakukan untuk fitur CAV lantai dasar &amp; atap.</li>
                  <li>Hanya 2 fitur (CAV, RCAV) — sensitif terhadap penempatan sensor dan tidak menangkap arah/torsi/mode getar lain atau lokasi kerusakan (paper asli juga memodelkan lokasi kerusakan per lantai — tidak diimplementasikan di sini).</li>
                  <li>RCAV tidak selalu turun monoton terhadap kerusakan — paper mencatat sebagian rekaman kecil dapat menaikkan RCAV akibat efek resonansi pada fase kerusakan awal.</li>
                  <li>Perlu data kerusakan riil (tagging ATC-20 / survei lapangan berpasangan dengan rekaman akselerasi) untuk kalibrasi θ dan β yang valid secara statistik, idealnya dengan pendekatan fragility-based threshold.</li>
                </ul>
              </div>
              <div>
                <div style={{ color: C.textPrimary }} className="font-medium mb-1">Kedudukan terhadap evaluasi formal</div>
                <p>Alat ini adalah skrining <i>cepat, berbasis sensor</i> untuk triase awal, melengkapi — bukan menggantikan — prosedur evaluasi &amp; rehabilitasi seismik formal (mis. SNI 9274:2025 / ASCE 41-17 Tier 1–3) yang tetap diperlukan untuk keputusan perbaikan atau pembongkaran.</p>
              </div>
            </div>
          )}
        </Panel>

        <footer style={{ color: C.textFaint, borderTop: `1px solid ${C.line}` }} className="text-xs pt-4 pb-2 flex items-start gap-2">
          <Building2 size={13} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>Prototipe demonstrasi metodologi. Parameter OLR ilustratif — kalibrasi ulang dengan data kerusakan riil diperlukan sebelum penggunaan operasional.</span>
        </footer>
      </div>
    </div>
  );
}
