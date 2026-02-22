import { useState, useEffect, useRef, useMemo, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════════════════════
   ORACLE-Sol — "The Scientist's Journal"
   
   Aesthetic: Warm dark base with brass/gold accents, evoking a high-end
   research publication crossed with laboratory instrumentation.
   
   Typography: Instrument Serif (display) + Manrope (body) + Fira Code (mono)
   Palette: Warm blacks, brass gold accent, muted sage/rose for sol/insol
   Background: Subtle animated particle field (protein molecules in solution)
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── Deterministic mock predictor ───────────────────────────────────────────
function hashSeq(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function predict(seq) {
  const c = seq.toUpperCase().replace(/[^A-Z]/g, "");
  if (c.length < 5) return null;
  const h = hashSeq(c);
  const hydro = (c.match(/[VILMFWP]/g) || []).length / c.length;
  const charged = (c.match(/[DEKRH]/g) || []).length / c.length;
  const cys = (c.match(/C/g) || []).length / c.length;
  let base = 0.55 + charged * 0.8 - hydro * 0.6 - cys * 1.2 + ((h % 100) - 50) / 500;
  base = Math.max(0.03, Math.min(0.97, base));

  const res = [];
  for (let i = 0; i < c.length; i++) {
    const s = Math.max(0, i - 12), e = Math.min(c.length, i + 12);
    const w = c.slice(s, e);
    const lh = (w.match(/[VILMFWP]/g) || []).length / w.length;
    const lc = (w.match(/[DEKRH]/g) || []).length / w.length;
    res.push(Math.max(0, Math.min(1, 0.5 + lc * 0.7 - lh * 0.5 + ((hashSeq(w) % 40) - 20) / 200)));
  }

  const conf = Math.abs(base - 0.5) * 2;
  return {
    score: base, label: base >= 0.5 ? "soluble" : "insoluble",
    confidence: conf >= 0.6 ? "high" : conf >= 0.3 ? "medium" : "low",
    length: c.length, truncated: c.length > 1022, residueScores: res,
  };
}

const REFS = [
  { name: "GFP", org: "A. victoria", seq: "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK", known: true },
  { name: "Lysozyme", org: "G. gallus", seq: "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", known: true },
  { name: "Thioredoxin", org: "E. coli", seq: "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA", known: true },
  { name: "MBP", org: "E. coli", seq: "MKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDIIFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNKDLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIKDVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNK", known: true },
  { name: "SUMO1", org: "H. sapiens", seq: "MSDQEAKPSTEDLGDKKEGEYIKLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPMNSLRFLLFEGQRIADNHTPKELGMEEEDVIEVYQEQTGGHSTV", known: true },
  { name: "Insulin", org: "H. sapiens", seq: "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN", known: false },
  { name: "p53", org: "H. sapiens", seq: "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLPGRNSFEV", known: false },
  { name: "Amyloid-beta 42", org: "H. sapiens", seq: "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA", known: false },
];

// ─── Ambient particle canvas ────────────────────────────────────────────────
function ParticleField() {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let raf;
    let particles = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: Math.random() * 1.5 + 0.3,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.1,
        opacity: Math.random() * 0.25 + 0.05,
        phase: Math.random() * Math.PI * 2,
      });
    }

    const draw = (t) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -10) p.x = canvas.width + 10;
        if (p.x > canvas.width + 10) p.x = -10;
        if (p.y < -10) p.y = canvas.height + 10;
        if (p.y > canvas.height + 10) p.y = -10;
        const flicker = 0.5 + 0.5 * Math.sin(t * 0.001 + p.phase);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200, 168, 124, ${p.opacity * flicker})`;
        ctx.fill();
      }

      // Draw faint connections between close particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(200, 168, 124, ${0.04 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none" }}
    />
  );
}

// ─── Animated number counter ────────────────────────────────────────────────
function Counter({ value, duration = 1400 }) {
  const [disp, setDisp] = useState(0);
  useEffect(() => {
    let start = null, raf;
    const run = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 4);
      setDisp(value * eased);
      if (p < 1) raf = requestAnimationFrame(run);
    };
    raf = requestAnimationFrame(run);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return <>{(disp * 100).toFixed(1)}</>;
}

// ─── Heatmap strip ──────────────────────────────────────────────────────────
function HeatStrip({ scores, length }) {
  const ref = useRef(null);
  const [w, setW] = useState(500);
  const [hov, setHov] = useState(null);

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const display = useMemo(() => {
    if (!scores?.length) return [];
    const max = Math.floor(w / 3);
    if (scores.length <= max) return scores;
    const r = Math.ceil(scores.length / max);
    const d = [];
    for (let i = 0; i < scores.length; i += r) {
      const chunk = scores.slice(i, i + r);
      d.push(chunk.reduce((a, b) => a + b, 0) / chunk.length);
    }
    return d;
  }, [scores, w]);

  const color = (v) => {
    if (v < 0.3) return "rgba(217, 116, 116, 0.9)";
    if (v < 0.5) return "rgba(234, 179, 8, 0.7)";
    if (v < 0.7) return "rgba(200, 168, 124, 0.6)";
    return "rgba(110, 200, 160, 0.85)";
  };

  if (!scores?.length) return null;

  return (
    <div ref={ref} style={{ width: "100%" }}>
      <div style={{
        display: "flex", justifyContent: "space-between", marginBottom: 12,
        fontFamily: "'Manrope', sans-serif", fontSize: 11, letterSpacing: "0.04em",
      }}>
        <span style={{ color: "var(--text-tertiary)", textTransform: "uppercase" }}>Residue contributions</span>
        {hov !== null && (
          <span style={{ color: "var(--text-secondary)" }}>
            ~position {Math.round((hov / display.length) * length)} of {length}
          </span>
        )}
      </div>
      <div style={{ display: "flex", gap: 1, height: 36, alignItems: "flex-end", cursor: "crosshair" }}>
        {display.map((s, i) => (
          <div
            key={i}
            onMouseEnter={() => setHov(i)}
            onMouseLeave={() => setHov(null)}
            style={{
              flex: 1, minWidth: 1, borderRadius: "2px 2px 0 0",
              height: `${30 + s * 70}%`,
              backgroundColor: color(s),
              opacity: hov === null ? 0.8 : hov === i ? 1 : 0.3,
              transition: "opacity 0.15s ease, height 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
            }}
          />
        ))}
      </div>
      <div style={{
        display: "flex", gap: 20, marginTop: 14,
        fontFamily: "'Manrope', sans-serif", fontSize: 10, color: "var(--text-tertiary)",
      }}>
        {[
          ["rgba(217,116,116,0.9)", "Destabilizing"],
          ["rgba(234,179,8,0.7)", "Neutral"],
          ["rgba(110,200,160,0.85)", "Stabilizing"],
        ].map(([c, l]) => (
          <div key={l} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: c }} />
            <span>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Comparison chart ───────────────────────────────────────────────────────
function Comparison({ userResult, refs }) {
  const all = useMemo(() => {
    const items = refs.map((r) => {
      const p = predict(r.seq);
      return { name: r.name, score: p.score, isUser: false, known: r.known };
    });
    if (userResult) items.push({ name: "Your protein", score: userResult.score, isUser: true });
    return items.sort((a, b) => b.score - a.score);
  }, [userResult, refs]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {all.map((item, i) => (
        <div
          key={item.name}
          style={{
            display: "grid", gridTemplateColumns: "140px 1fr 56px",
            alignItems: "center", gap: 16, padding: "8px 0",
            opacity: 0, animation: `revealUp 0.45s cubic-bezier(0.16,1,0.3,1) ${i * 55}ms forwards`,
            borderBottom: item.isUser ? "none" : "1px solid rgba(200,168,124,0.04)",
          }}
        >
          <span style={{
            fontFamily: item.isUser ? "'Instrument Serif', serif" : "'Manrope', sans-serif",
            fontSize: item.isUser ? 14 : 12,
            fontStyle: item.isUser ? "italic" : "normal",
            color: item.isUser ? "var(--accent)" : "var(--text-tertiary)",
            textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {item.name}
          </span>
          <div style={{
            height: item.isUser ? 24 : 10,
            borderRadius: 99,
            backgroundColor: "rgba(200,168,124,0.06)",
            overflow: "hidden",
            transition: "height 0.3s cubic-bezier(0.16,1,0.3,1)",
          }}>
            <div style={{
              height: "100%", borderRadius: 99,
              width: `${item.score * 100}%`,
              background: item.isUser
                ? "linear-gradient(90deg, var(--accent), #e0c99a)"
                : item.score >= 0.5
                  ? "rgba(110,200,160,0.35)"
                  : "rgba(217,116,116,0.3)",
              transition: "width 0.9s cubic-bezier(0.16,1,0.3,1)",
              ...(item.isUser ? { boxShadow: "0 0 24px rgba(200,168,124,0.2)" } : {}),
            }} />
          </div>
          <span style={{
            fontFamily: "'Fira Code', monospace", fontSize: 11,
            fontWeight: item.isUser ? 600 : 400,
            color: item.isUser ? "var(--accent)"
              : item.score >= 0.5 ? "var(--soluble)" : "var(--insoluble)",
          }}>
            {(item.score * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Batch table ────────────────────────────────────────────────────────────
function BatchTable({ results }) {
  if (!results?.length) return null;
  const sorted = [...results].sort((a, b) => b.score - a.score);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            {["Rank", "Protein", "Length", "Score", "Verdict", ""].map((h, i) => (
              <th key={h} style={{
                padding: "14px 16px",
                textAlign: i === 3 ? "right" : "left",
                fontFamily: "'Manrope', sans-serif",
                fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase",
                color: "var(--text-tertiary)",
                borderBottom: "1px solid rgba(200,168,124,0.08)",
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} style={{
              opacity: 0,
              animation: `revealUp 0.35s cubic-bezier(0.16,1,0.3,1) ${i * 45}ms forwards`,
            }}>
              <td style={{ padding: "14px 16px", fontFamily: "'Fira Code', monospace", fontSize: 12, color: "var(--text-tertiary)" }}>{String(i + 1).padStart(2, "0")}</td>
              <td style={{ padding: "14px 16px", fontFamily: "'Manrope', sans-serif", fontSize: 13, fontWeight: 500, color: "var(--text-primary)" }}>{r.name}</td>
              <td style={{ padding: "14px 16px", fontFamily: "'Fira Code', monospace", fontSize: 12, color: "var(--text-tertiary)" }}>{r.length}</td>
              <td style={{
                padding: "14px 16px", textAlign: "right",
                fontFamily: "'Fira Code', monospace", fontSize: 13, fontWeight: 600,
                color: r.score >= 0.5 ? "var(--soluble)" : "var(--insoluble)",
              }}>{(r.score * 100).toFixed(1)}%</td>
              <td style={{ padding: "14px 16px" }}>
                <span style={{
                  display: "inline-block", padding: "3px 10px", borderRadius: 99,
                  fontSize: 10, fontFamily: "'Manrope', sans-serif", fontWeight: 600, letterSpacing: "0.04em",
                  textTransform: "uppercase",
                  color: r.score >= 0.5 ? "var(--soluble)" : "var(--insoluble)",
                  backgroundColor: r.score >= 0.5 ? "rgba(110,200,160,0.1)" : "rgba(217,116,116,0.1)",
                  border: `1px solid ${r.score >= 0.5 ? "rgba(110,200,160,0.15)" : "rgba(217,116,116,0.15)"}`,
                }}>
                  {r.label}
                </span>
              </td>
              <td style={{ padding: "14px 16px", width: 100 }}>
                <div style={{ display: "flex", gap: 2 }}>
                  {Array.from({ length: 10 }).map((_, j) => (
                    <div key={j} style={{
                      width: 6, height: 6, borderRadius: 1,
                      backgroundColor: j / 10 < r.score
                        ? (r.score >= 0.5 ? "var(--soluble)" : "var(--insoluble)")
                        : "rgba(200,168,124,0.08)",
                      opacity: j / 10 < r.score ? 0.7 : 1,
                    }} />
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Main ───────────────────────────────────────────────────────────────────
export default function OracleSol() {
  const [input, setInput] = useState("");
  const [view, setView] = useState("idle"); // idle | single | batch | compare
  const [result, setResult] = useState(null);
  const [batchResults, setBatch] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [key, setKey] = useState(0); // force re-mount for animations

  const run = useCallback(() => {
    const t = input.trim();
    if (!t) { setError("Paste a protein sequence or FASTA to begin."); return; }
    setError(null); setProcessing(true); setResult(null); setBatch(null);

    setTimeout(() => {
      try {
        if (t.startsWith(">")) {
          const entries = [];
          let cn = "", cs = [];
          for (const line of t.split("\n")) {
            if (line.startsWith(">")) {
              if (cn) entries.push({ name: cn, seq: cs.join("") });
              cn = line.slice(1).split(/\s+/)[0] || `seq_${entries.length + 1}`;
              cs = [];
            } else cs.push(line.trim());
          }
          if (cn) entries.push({ name: cn, seq: cs.join("") });

          if (entries.length > 1) {
            const r = entries.map(e => { const p = predict(e.seq); return p ? { ...p, name: e.name } : null; }).filter(Boolean);
            setBatch(r); setView("batch"); setKey(k => k + 1);
          } else if (entries.length === 1) {
            const p = predict(entries[0].seq);
            if (p) { setResult({ ...p, name: entries[0].name }); setView("single"); setKey(k => k + 1); }
            else setError("Sequence too short (min 5 residues).");
          }
        } else {
          const p = predict(t);
          if (p) { setResult({ ...p, name: "query" }); setView("single"); setKey(k => k + 1); }
          else setError("Sequence too short (min 5 residues).");
        }
      } catch { setError("Could not parse input."); }
      setProcessing(false);
    }, 900);
  }, [input]);

  const loadEx = (type) => {
    if (type === "gfp") setInput("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK");
    else setInput(`>GFP_avictoria\nMSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL\nVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN\nRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY\n>Insulin_human\nMALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT\n>TrxA_ecoli\nMSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGK\n>DesignA_mpnn\nMKFLILLFNILCLFPVLAADYKDDDDKGASVDKEIRSALNLMIRDGKLVLNRGK\n>AmyloidBeta42\nDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA`);
  };

  const hasResults = view !== "idle";
  const labelColor = result?.score >= 0.5 ? "var(--soluble)" : "var(--insoluble)";

  return (
    <div style={{
      "--accent": "#c8a87c",
      "--accent-glow": "rgba(200,168,124,0.15)",
      "--soluble": "#6ec8a0",
      "--insoluble": "#d97474",
      "--bg": "#09090b",
      "--surface": "#111113",
      "--surface-raised": "#19191c",
      "--border": "rgba(200,168,124,0.08)",
      "--text-primary": "#e8e5df",
      "--text-secondary": "#9a958d",
      "--text-tertiary": "#5c5850",
      minHeight: "100dvh",
      backgroundColor: "var(--bg)",
      color: "var(--text-primary)",
      fontFamily: "'Manrope', sans-serif",
      position: "relative",
      overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');

        @keyframes revealUp {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes revealScale {
          from { opacity: 0; transform: scale(0.96); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes fadeIn {
          to { opacity: 1; }
        }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        @keyframes breathe {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        @keyframes counterPulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.02); }
          100% { transform: scale(1); }
        }

        textarea::placeholder { color: rgba(200,168,124,0.2); }
        textarea:focus { outline: none; border-color: rgba(200,168,124,0.25) !important; }
        textarea { scrollbar-width: thin; scrollbar-color: rgba(200,168,124,0.1) transparent; }
        button:active { transform: translateY(1px) !important; }

        .oracle-main { transition: all 0.6s cubic-bezier(0.16,1,0.3,1); }

        @media (max-width: 768px) {
          .oracle-split { grid-template-columns: 1fr !important; }
          .oracle-metrics { grid-template-columns: 1fr 1fr !important; }
          .oracle-score-display { font-size: 64px !important; }
        }
      `}</style>

      {/* Ambient background */}
      <ParticleField />

      {/* Warm radial glow */}
      <div style={{
        position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none",
        background: "radial-gradient(ellipse 60% 50% at 30% 20%, rgba(200,168,124,0.03), transparent 70%), radial-gradient(ellipse 40% 60% at 80% 80%, rgba(110,200,160,0.02), transparent 60%)",
      }} />

      {/* Grain overlay */}
      <div style={{
        position: "fixed", inset: 0, zIndex: 1, pointerEvents: "none", opacity: 0.35,
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E")`,
      }} />

      <div style={{ position: "relative", zIndex: 2 }}>

        {/* ─── Navigation ─────────────────────────────── */}
        <nav style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "20px 40px", borderBottom: "1px solid var(--border)",
          opacity: 0, animation: "revealUp 0.6s cubic-bezier(0.16,1,0.3,1) 0.1s forwards",
        }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={{
              fontFamily: "'Instrument Serif', serif", fontStyle: "italic",
              fontSize: 22, color: "var(--accent)", letterSpacing: "-0.02em",
            }}>Oracle</span>
            <span style={{
              fontFamily: "'Manrope', sans-serif", fontSize: 11, fontWeight: 300,
              color: "var(--text-tertiary)", letterSpacing: "0.12em", textTransform: "uppercase",
            }}>Sol</span>
          </div>
          <div style={{
            display: "flex", alignItems: "center", gap: 20,
            fontFamily: "'Fira Code', monospace", fontSize: 10, color: "var(--text-tertiary)",
          }}>
            <span>ESM2-650M</span>
            <span style={{ width: 3, height: 3, borderRadius: "50%", backgroundColor: "var(--text-tertiary)", opacity: 0.4 }} />
            <span>73.4% acc</span>
            <span style={{ width: 3, height: 3, borderRadius: "50%", backgroundColor: "var(--text-tertiary)", opacity: 0.4 }} />
            <span>MCC 0.455</span>
          </div>
        </nav>

        {/* ─── Main content ───────────────────────────── */}
        <div className="oracle-main" style={{
          maxWidth: hasResults ? "none" : 620,
          margin: hasResults ? 0 : "0 auto",
          padding: hasResults ? 0 : "0 40px",
        }}>

          {/* ─── Idle: centered hero ─────────────────── */}
          {!hasResults && (
            <div style={{
              display: "flex", flexDirection: "column", justifyContent: "center",
              minHeight: "calc(100dvh - 70px)", paddingBottom: 80,
            }}>
              <div style={{
                marginBottom: 48,
                opacity: 0, animation: "revealUp 0.7s cubic-bezier(0.16,1,0.3,1) 0.25s forwards",
              }}>
                <h1 style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontSize: 48, fontWeight: 400, lineHeight: 1.1,
                  letterSpacing: "-0.03em", color: "var(--text-primary)", marginBottom: 16,
                }}>
                  Will your protein<br />
                  <span style={{ fontStyle: "italic", color: "var(--accent)" }}>express?</span>
                </h1>
                <p style={{
                  fontFamily: "'Manrope', sans-serif",
                  fontSize: 15, fontWeight: 300, lineHeight: 1.7,
                  color: "var(--text-secondary)", maxWidth: "48ch",
                }}>
                  Predict E. coli solubility from sequence alone. Frozen ESM2 embeddings
                  scored against 70,000 experimentally validated proteins.
                </p>
              </div>

              {/* Input area */}
              <div style={{
                opacity: 0, animation: "revealUp 0.7s cubic-bezier(0.16,1,0.3,1) 0.4s forwards",
              }}>
                <textarea
                  value={input}
                  onChange={(e) => { setInput(e.target.value); setError(null); }}
                  onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); }}
                  placeholder="Paste amino acid sequence or FASTA..."
                  spellCheck={false}
                  rows={5}
                  style={{
                    width: "100%", padding: "20px 22px",
                    backgroundColor: "var(--surface)",
                    border: error ? "1px solid rgba(217,116,116,0.4)" : "1px solid var(--border)",
                    borderRadius: 14, color: "var(--text-primary)",
                    fontFamily: "'Fira Code', monospace", fontSize: 13, lineHeight: 1.7,
                    resize: "vertical",
                    transition: "border-color 0.3s cubic-bezier(0.16,1,0.3,1)",
                  }}
                />
                {error && <p style={{ color: "var(--insoluble)", fontSize: 12, marginTop: 8 }}>{error}</p>}
              </div>

              {/* Example chips + predict button */}
              <div style={{
                display: "flex", alignItems: "center", gap: 12, marginTop: 16, flexWrap: "wrap",
                opacity: 0, animation: "revealUp 0.6s cubic-bezier(0.16,1,0.3,1) 0.55s forwards",
              }}>
                <button onClick={run} disabled={processing || !input.trim()} style={{
                  padding: "12px 28px", borderRadius: 99, border: "none",
                  background: input.trim()
                    ? "linear-gradient(135deg, #b8944f, #c8a87c)"
                    : "rgba(200,168,124,0.08)",
                  color: input.trim() ? "#09090b" : "var(--text-tertiary)",
                  fontFamily: "'Manrope', sans-serif", fontSize: 13, fontWeight: 600,
                  cursor: input.trim() ? "pointer" : "default",
                  transition: "all 0.3s cubic-bezier(0.16,1,0.3,1)",
                  position: "relative", overflow: "hidden",
                }}>
                  {processing ? (
                    <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ animation: "breathe 1.2s ease infinite" }}>{"\u25C7"}</span>
                      Scoring...
                    </span>
                  ) : "Predict"}
                  {processing && (
                    <div style={{
                      position: "absolute", inset: 0,
                      background: "linear-gradient(90deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%)",
                      backgroundSize: "200% 100%",
                      animation: "shimmer 1.5s infinite",
                    }} />
                  )}
                </button>

                <div style={{ width: 1, height: 20, backgroundColor: "var(--border)" }} />

                {[["gfp", "GFP sequence"], ["batch", "Batch FASTA"]].map(([k, label]) => (
                  <button key={k} onClick={() => loadEx(k)} style={{
                    padding: "8px 16px", borderRadius: 99, fontSize: 11,
                    border: "1px solid var(--border)", background: "transparent",
                    color: "var(--text-tertiary)", cursor: "pointer", fontFamily: "'Manrope', sans-serif",
                    transition: "all 0.25s cubic-bezier(0.16,1,0.3,1)",
                  }}
                    onMouseEnter={(e) => { e.target.style.borderColor = "rgba(200,168,124,0.3)"; e.target.style.color = "var(--accent)"; }}
                    onMouseLeave={(e) => { e.target.style.borderColor = "var(--border)"; e.target.style.color = "var(--text-tertiary)"; }}
                  >{label}</button>
                ))}
              </div>

              <div style={{
                marginTop: 14, fontSize: 10, fontFamily: "'Fira Code', monospace",
                color: "var(--text-tertiary)", opacity: 0.5,
              }}>
                {"\u2318"} + Enter to predict
              </div>
            </div>
          )}

          {/* ─── Results: split layout ───────────────── */}
          {hasResults && (
            <div className="oracle-split" style={{
              display: "grid", gridTemplateColumns: "340px 1fr",
              minHeight: "calc(100dvh - 70px)",
            }}>

              {/* Left sidebar: input + tabs */}
              <div style={{
                padding: "36px 32px",
                borderRight: "1px solid var(--border)",
                display: "flex", flexDirection: "column", gap: 20,
                opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s forwards",
              }}>
                <span style={{
                  fontFamily: "'Instrument Serif', serif", fontStyle: "italic",
                  fontSize: 13, color: "var(--text-tertiary)",
                }}>Input sequence</span>

                <textarea
                  value={input}
                  onChange={(e) => { setInput(e.target.value); setError(null); }}
                  onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); }}
                  spellCheck={false}
                  rows={6}
                  style={{
                    width: "100%", padding: "14px 16px",
                    backgroundColor: "var(--surface)",
                    border: "1px solid var(--border)", borderRadius: 10,
                    color: "var(--text-primary)",
                    fontFamily: "'Fira Code', monospace", fontSize: 11, lineHeight: 1.6,
                    resize: "vertical",
                  }}
                />

                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={run} disabled={processing} style={{
                    flex: 1, padding: "10px 0", borderRadius: 8, border: "none",
                    background: "linear-gradient(135deg, #b8944f, #c8a87c)",
                    color: "#09090b", fontFamily: "'Manrope', sans-serif",
                    fontSize: 12, fontWeight: 600, cursor: "pointer",
                    transition: "all 0.2s",
                  }}>
                    {processing ? "Scoring..." : "Re-predict"}
                  </button>
                </div>

                {/* View tabs */}
                <div style={{
                  display: "flex", flexDirection: "column", gap: 2,
                  borderTop: "1px solid var(--border)", paddingTop: 20, marginTop: 8,
                }}>
                  <span style={{
                    fontFamily: "'Instrument Serif', serif", fontStyle: "italic",
                    fontSize: 13, color: "var(--text-tertiary)", marginBottom: 8,
                  }}>View</span>
                  {[
                    ["single", "Detail"],
                    ...(batchResults ? [["batch", "Ranked table"]] : []),
                    ["compare", "Reference comparison"],
                  ].map(([v, label]) => (
                    <button key={v} onClick={() => { setView(v); setKey(k => k + 1); }} style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "10px 14px", borderRadius: 8, border: "none",
                      background: view === v ? "rgba(200,168,124,0.08)" : "transparent",
                      cursor: "pointer", textAlign: "left",
                      transition: "all 0.2s cubic-bezier(0.16,1,0.3,1)",
                    }}>
                      <div style={{
                        width: 5, height: 5, borderRadius: "50%",
                        backgroundColor: view === v ? "var(--accent)" : "transparent",
                        border: view === v ? "none" : "1px solid var(--text-tertiary)",
                        transition: "all 0.2s",
                      }} />
                      <span style={{
                        fontFamily: "'Manrope', sans-serif", fontSize: 12,
                        color: view === v ? "var(--accent)" : "var(--text-secondary)",
                        fontWeight: view === v ? 500 : 400,
                      }}>{label}</span>
                    </button>
                  ))}
                </div>

                {/* Footnote */}
                <div style={{
                  marginTop: "auto", paddingTop: 20, borderTop: "1px solid var(--border)",
                  fontSize: 10, lineHeight: 1.7, color: "var(--text-tertiary)",
                  fontFamily: "'Manrope', sans-serif",
                }}>
                  Accuracy ceiling ~77% across all methods, limited by label noise in TargetTrack experimental data.
                </div>
              </div>

              {/* Right: result content */}
              <div key={key} style={{ padding: "48px 56px", overflowY: "auto" }}>

                {/* ── Single detail ──────────────────── */}
                {view === "single" && result && (
                  <div>
                    {/* Big score */}
                    <div style={{
                      display: "flex", alignItems: "flex-end", gap: 16, marginBottom: 12,
                      opacity: 0, animation: "revealScale 0.8s cubic-bezier(0.16,1,0.3,1) 0.15s forwards",
                    }}>
                      <span className="oracle-score-display" style={{
                        fontFamily: "'Instrument Serif', serif",
                        fontSize: 96, fontWeight: 400, lineHeight: 0.85,
                        color: labelColor, letterSpacing: "-0.04em",
                      }}>
                        <Counter value={result.score} />
                      </span>
                      <span style={{
                        fontFamily: "'Fira Code', monospace",
                        fontSize: 18, color: labelColor, opacity: 0.5,
                        marginBottom: 12,
                      }}>%</span>
                    </div>

                    {/* Label */}
                    <div style={{
                      marginBottom: 40,
                      opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.4s forwards",
                    }}>
                      <span style={{
                        fontFamily: "'Instrument Serif', serif", fontStyle: "italic",
                        fontSize: 22, color: labelColor,
                      }}>
                        Predicted {result.label}
                      </span>
                      <span style={{
                        display: "inline-block", marginLeft: 14,
                        padding: "3px 10px", borderRadius: 99,
                        fontSize: 10, fontFamily: "'Manrope', sans-serif",
                        fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase",
                        color: result.confidence === "high" ? "var(--soluble)"
                          : result.confidence === "medium" ? "var(--accent)" : "var(--insoluble)",
                        backgroundColor: result.confidence === "high" ? "rgba(110,200,160,0.1)"
                          : result.confidence === "medium" ? "rgba(200,168,124,0.1)" : "rgba(217,116,116,0.1)",
                        border: `1px solid ${
                          result.confidence === "high" ? "rgba(110,200,160,0.15)"
                          : result.confidence === "medium" ? "rgba(200,168,124,0.15)" : "rgba(217,116,116,0.15)"
                        }`,
                      }}>
                        {result.confidence} confidence
                      </span>
                    </div>

                    {/* Metrics row */}
                    <div className="oracle-metrics" style={{
                      display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 0,
                      borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)",
                      marginBottom: 48,
                      opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.55s forwards",
                    }}>
                      {[
                        ["Sequence length", `${result.length} aa`],
                        ["Prediction", result.label],
                        ["Score", (result.score * 100).toFixed(1) + "%"],
                        ["Truncated", result.truncated ? "Yes (>1022)" : "No"],
                      ].map(([label, val], i) => (
                        <div key={label} style={{
                          padding: "20px 0",
                          borderRight: i < 3 ? "1px solid var(--border)" : "none",
                          paddingLeft: i > 0 ? 24 : 0,
                        }}>
                          <div style={{
                            fontSize: 10, fontFamily: "'Manrope', sans-serif",
                            fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase",
                            color: "var(--text-tertiary)", marginBottom: 6,
                          }}>{label}</div>
                          <div style={{
                            fontSize: 14, fontFamily: "'Fira Code', monospace",
                            fontWeight: 500, color: "var(--text-primary)",
                          }}>{val}</div>
                        </div>
                      ))}
                    </div>

                    {/* Heatmap */}
                    <div style={{
                      opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.7s forwards",
                    }}>
                      <HeatStrip scores={result.residueScores} length={result.length} />
                    </div>
                  </div>
                )}

                {/* ── Batch table ────────────────────── */}
                {view === "batch" && batchResults && (
                  <div>
                    <div style={{
                      display: "flex", justifyContent: "space-between", alignItems: "baseline",
                      marginBottom: 32,
                      opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s forwards",
                    }}>
                      <h2 style={{
                        fontFamily: "'Instrument Serif', serif", fontSize: 28, fontWeight: 400,
                        letterSpacing: "-0.02em",
                      }}>Ranked results</h2>
                      <span style={{
                        fontFamily: "'Fira Code', monospace", fontSize: 12, color: "var(--text-tertiary)",
                      }}>{batchResults.length} sequences</span>
                    </div>
                    <BatchTable results={batchResults} />

                    {/* Summary strip */}
                    <div style={{
                      display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 0,
                      marginTop: 32, borderTop: "1px solid var(--border)", paddingTop: 24,
                      opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.4s forwards",
                    }}>
                      {[
                        ["Total", batchResults.length, "var(--text-primary)"],
                        ["Soluble", batchResults.filter(r => r.label === "soluble").length, "var(--soluble)"],
                        ["Insoluble", batchResults.filter(r => r.label === "insoluble").length, "var(--insoluble)"],
                        ["High conf.", batchResults.filter(r => r.confidence === "high").length, "var(--accent)"],
                      ].map(([label, val, color]) => (
                        <div key={label}>
                          <div style={{
                            fontSize: 10, fontFamily: "'Manrope', sans-serif",
                            fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase",
                            color: "var(--text-tertiary)", marginBottom: 8,
                          }}>{label}</div>
                          <div style={{
                            fontFamily: "'Instrument Serif', serif",
                            fontSize: 32, fontWeight: 400, color,
                          }}>{val}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Compare ────────────────────────── */}
                {view === "compare" && (
                  <div>
                    <div style={{
                      marginBottom: 36,
                      opacity: 0, animation: "revealUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s forwards",
                    }}>
                      <h2 style={{
                        fontFamily: "'Instrument Serif', serif", fontSize: 28, fontWeight: 400,
                        letterSpacing: "-0.02em", marginBottom: 8,
                      }}>Reference comparison</h2>
                      <p style={{
                        fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7,
                        fontWeight: 300, maxWidth: "52ch",
                      }}>
                        Your protein scored against well-characterized E. coli expression benchmarks,
                        from highly soluble fusion tags to notorious aggregation-prone peptides.
                      </p>
                    </div>
                    <Comparison userResult={result} refs={REFS} />
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
