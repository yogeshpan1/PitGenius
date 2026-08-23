import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [races, setRaces] = useState([]);
  const [sel, setSel] = useState(null);
  const [results, setResults] = useState([]);

  useEffect(() => {
    fetch(`${API}/api/races`).then(r => r.json()).then(setRaces).catch(console.error);
  }, []);

  useEffect(() => {
    if (!sel) return;
    fetch(`${API}/api/races/${sel.year}/${sel.round}/results`)
      .then(r => r.json()).then(setResults).catch(console.error);
  }, [sel]);

  return (
    <div style={{ fontFamily: "system-ui", margin: "2rem auto", maxWidth: 900 }}>
      <h1>🏁 PitGenius</h1>
      <p>F1 race strategy analytics — historical explorer (production frontend).</p>
      <select onChange={e => setSel(JSON.parse(e.target.value))} defaultValue="">
        <option value="" disabled>Select a race…</option>
        {races.map((r, i) => (
          <option key={i} value={JSON.stringify(r)}>
            {r.year} R{r.round} — {r.event_name}
          </option>
        ))}
      </select>
      {results.length > 0 && (
        <table style={{ marginTop: "1rem", borderCollapse: "collapse" }} border="1">
          <thead><tr><th>Pos</th><th>Driver</th><th>Team</th><th>Status</th><th>Pts</th></tr></thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i}><td>{r.position ?? "DNF"}</td><td>{r.driver}</td>
                <td>{r.team}</td><td>{r.status}</td><td>{r.points}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}