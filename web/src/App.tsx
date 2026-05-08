import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import explainerText from "./explainer.md?raw";
import type { ConfigKey, Entry, Metadata } from "./types";

const CONFIGS: ConfigKey[] = ["XMglass", "DJI", "Multiview"];

function useMetadata() {
  const [data, setData] = useState<Metadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch("/metadata.json")
      .then((r) => {
        if (!r.ok) throw new Error(`metadata fetch ${r.status}`);
        return r.json() as Promise<Metadata>;
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);
  return { data, error };
}

function classNames(...xs: (string | false | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

interface Filters {
  configs: Set<ConfigKey>;
  scene: string;
  onlyProtocol: boolean;
  onlyIssue: boolean;
  search: string;
}

function applyFilters(entries: Entry[], f: Filters): Entry[] {
  const search = f.search.trim().toLowerCase();
  return entries.filter((e) => {
    if (!f.configs.has(e.config)) return false;
    if (f.scene && e.scene !== f.scene) return false;
    if (f.onlyProtocol && !e.has_protocol) return false;
    if (f.onlyIssue && !e.issue) return false;
    if (search) {
      const hay = `${e.slice_id} ${e.operation} ${e.scene} ${e.protocol_name} ${e.video_name}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });
}

function Sidebar({
  entries,
  filters,
  setFilters,
  selectedId,
  onSelect,
  scenes,
}: {
  entries: Entry[];
  filters: Filters;
  setFilters: (f: Filters) => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
  scenes: string[];
}) {
  const toggleConfig = (cfg: ConfigKey) => {
    const next = new Set(filters.configs);
    if (next.has(cfg)) next.delete(cfg);
    else next.add(cfg);
    setFilters({ ...filters, configs: next });
  };

  return (
    <aside className="sidebar">
      <div className="filters">
        <input
          type="search"
          placeholder="search id, operation, scene…"
          value={filters.search}
          onChange={(e) => setFilters({ ...filters, search: e.target.value })}
        />
        <div className="filter-row">
          {CONFIGS.map((cfg) => (
            <label key={cfg} className={classNames("chip", filters.configs.has(cfg) && "chip--on")}>
              <input
                type="checkbox"
                checked={filters.configs.has(cfg)}
                onChange={() => toggleConfig(cfg)}
              />
              {cfg}
            </label>
          ))}
        </div>
        <div className="filter-row">
          <select
            value={filters.scene}
            onChange={(e) => setFilters({ ...filters, scene: e.target.value })}
          >
            <option value="">all scenes</option>
            {scenes.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div className="filter-row">
          <label className="check">
            <input
              type="checkbox"
              checked={filters.onlyProtocol}
              onChange={(e) => setFilters({ ...filters, onlyProtocol: e.target.checked })}
            />
            has protocol
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={filters.onlyIssue}
              onChange={(e) => setFilters({ ...filters, onlyIssue: e.target.checked })}
            />
            has issue
          </label>
        </div>
        <div className="count">{entries.length} entries</div>
      </div>
      <ul className="entry-list">
        {entries.map((e) => (
          <li
            key={e.slice_id}
            className={classNames("entry-row", selectedId === e.slice_id && "entry-row--active")}
            onClick={() => onSelect(e.slice_id)}
          >
            <div className="entry-row__top">
              <span className="entry-id">{e.slice_id}</span>
              <span className={classNames("badge", `badge--${e.config.toLowerCase()}`)}>
                {e.config}
              </span>
            </div>
            <div className="entry-row__op">
              {e.operation || <span className="muted">(no operation label)</span>}
            </div>
            <div className="entry-row__meta">
              <span>{e.scene || "—"}</span>
              <span>{e.length || "—"}</span>
              {e.has_protocol && <span className="tag tag--proto">proto</span>}
              {e.issue && <span className="tag tag--issue">issue</span>}
              {e.gpt4o_output && <span className="tag tag--gpt">gpt4o</span>}
              {!e.video_exists && <span className="tag tag--missing">missing</span>}
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}

function PathBlock({ path, exists }: { path: string; exists: boolean }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* ignore */
    }
  };
  return (
    <div className={classNames("pathblock", !exists && "pathblock--missing")}>
      <div className="pathblock__label">
        {exists ? "video file path" : "video file path (NOT FOUND on disk)"}
      </div>
      <div className="pathblock__row">
        <code className="pathblock__path">{path}</code>
        <button className="pathblock__copy" onClick={onCopy}>
          {copied ? "copied" : "copy"}
        </button>
      </div>
      {exists && (
        <div className="pathblock__hint">
          Open with <code>mpv</code>, <code>vlc</code>, or your file manager.
        </div>
      )}
    </div>
  );
}

function Detail({ entry, protocols }: { entry: Entry; protocols: Metadata["protocols"] }) {
  const protocolText = entry.has_protocol
    ? protocols[entry.config]?.[entry.protocol_name]
    : null;

  return (
    <section className="detail">
      <div className="detail__header">
        <h2>
          {entry.slice_id}
          <span className={classNames("badge", `badge--${entry.config.toLowerCase()}`)}>
            {entry.config}
          </span>
        </h2>
        <div className="detail__op">
          {entry.operation || <span className="muted">(no operation label)</span>}
        </div>
      </div>

      <PathBlock path={entry.video_path} exists={entry.video_exists} />

      <div className="meta-grid">
        <Meta label="scene" value={entry.scene} />
        <Meta label="length" value={entry.length} />
        <Meta label="date" value={entry.date} />
        <Meta label="video" value={entry.video_name} mono />
        <Meta label="protocol file" value={entry.protocol_name || "—"} mono />
        <Meta label="tools" value={entry.tools || "—"} />
        <Meta label="issue" value={entry.issue || "—"} highlight={!!entry.issue} />
        <Meta label="timestamps" value={entry.time_stamp || "—"} mono />
      </div>

      <div className="panes">
        <div className="pane">
          <div className="pane__header">Gold-standard protocol</div>
          {protocolText ? (
            <pre className="pane__body">{protocolText}</pre>
          ) : (
            <div className="pane__body pane__body--muted">
              No protocol annotated for this entry.
            </div>
          )}
        </div>

        {entry.gpt4o_output && (
          <div className="pane">
            <div className="pane__header">GPT-4o output (from CSV)</div>
            <pre className="pane__body">{entry.gpt4o_output}</pre>
          </div>
        )}
      </div>
    </section>
  );
}

function Meta({
  label,
  value,
  mono,
  highlight,
}: {
  label: string;
  value: string;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div className={classNames("meta-cell", highlight && "meta-cell--hl")}>
      <div className="meta-cell__label">{label}</div>
      <div className={classNames("meta-cell__value", mono && "meta-cell__value--mono")}>
        {value || "—"}
      </div>
    </div>
  );
}

function About() {
  return (
    <div className="about">
      <ReactMarkdown>{explainerText}</ReactMarkdown>
    </div>
  );
}

export default function App() {
  const { data, error } = useMetadata();
  const [tab, setTab] = useState<"browse" | "about">("browse");
  const [filters, setFilters] = useState<Filters>({
    configs: new Set(CONFIGS),
    scene: "",
    onlyProtocol: false,
    onlyIssue: false,
    search: "",
  });
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const filtered = useMemo(
    () => (data ? applyFilters(data.entries, filters) : []),
    [data, filters],
  );

  const scenes = useMemo(() => {
    if (!data) return [];
    const s = new Set<string>();
    for (const e of data.entries) if (e.scene) s.add(e.scene);
    return Array.from(s).sort();
  }, [data]);

  // Auto-select the first filtered row whenever the filter changes the head.
  useEffect(() => {
    if (filtered.length === 0) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !filtered.some((e) => e.slice_id === selectedId)) {
      setSelectedId(filtered[0].slice_id);
    }
  }, [filtered, selectedId]);

  const selected = data?.entries.find((e) => e.slice_id === selectedId) ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__title">LSV Viewer</div>
        <nav className="tabs">
          <button
            className={classNames("tab", tab === "browse" && "tab--on")}
            onClick={() => setTab("browse")}
          >
            Browse
          </button>
          <button
            className={classNames("tab", tab === "about" && "tab--on")}
            onClick={() => setTab("about")}
          >
            How frontier VLMs do this
          </button>
        </nav>
        {data && (
          <div className="topbar__stats">
            {data.entries.length} entries · {data.entries.filter((e) => e.has_protocol).length}{" "}
            with protocols
          </div>
        )}
      </header>

      {error && <div className="error">Failed to load metadata: {error}</div>}

      {tab === "browse" && data && (
        <main className="layout">
          <Sidebar
            entries={filtered}
            filters={filters}
            setFilters={setFilters}
            selectedId={selectedId}
            onSelect={setSelectedId}
            scenes={scenes}
          />
          {selected ? (
            <Detail entry={selected} protocols={data.protocols} />
          ) : (
            <section className="detail detail--empty">No entry selected.</section>
          )}
        </main>
      )}

      {tab === "about" && <About />}
    </div>
  );
}
