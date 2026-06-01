import { useState, useEffect } from 'react'

export default function EndToEndRuns() {
  const [runs, setRuns] = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [runData, setRunData] = useState(null)
  const [selectedClip, setSelectedClip] = useState(null)
  const [filter, setFilter] = useState('all')
  const [groupBy, setGroupBy] = useState('plate') // 'plate' or 'fold'

  // Load run list
  useEffect(() => {
    fetch('/api/e2e_runs').then(r => r.json()).then(data => {
      setRuns(data)
      if (data.length > 0 && !selectedRun) {
        setSelectedRun(data[0].name)
      }
    })
  }, [])

  // Load run data
  useEffect(() => {
    if (!selectedRun) return
    fetch(`/api/e2e_runs/${selectedRun}`).then(r => r.json()).then(data => {
      setRunData(data)
      setSelectedClip(null)
    })
  }, [selectedRun])

  const clips = runData?.clips || []
  const summary = runData?.summary || {}
  const plateBreakdown = summary.plate_breakdown || {}

  // Detect if this is a CV run (clips have fold field)
  const hasFolds = clips.length > 0 && clips[0].fold !== undefined

  // Filter clips
  const filteredClips = clips.filter(c => {
    if (filter === 'all') return true
    if (filter === 'correct') return c.correct
    if (filter === 'wrong') return !c.correct && c.status === 'ok'
    if (filter === 'fail') return c.status !== 'ok'
    return true
  })

  // Group clips by plate or fold
  const groupedClips = {}
  for (const clip of filteredClips) {
    const key = groupBy === 'fold' && hasFolds
      ? `Fold ${clip.fold}`
      : (clip.plate || 'Unknown')
    if (!groupedClips[key]) groupedClips[key] = []
    groupedClips[key].push(clip)
  }
  const groupKeys = Object.keys(groupedClips).sort((a, b) => {
    const na = parseInt(a.replace(/\D/g, '')) || 0
    const nb = parseInt(b.replace(/\D/g, '')) || 0
    return na - nb
  })

  const countByFilter = (f) => {
    if (f === 'all') return clips.length
    if (f === 'correct') return clips.filter(c => c.correct).length
    if (f === 'wrong') return clips.filter(c => !c.correct && c.status === 'ok').length
    if (f === 'fail') return clips.filter(c => c.status !== 'ok').length
    return 0
  }

  const badgeClass = (clip) => {
    if (clip.correct) return 'ok'
    if (clip.status !== 'ok') return 'fail'
    return 'miss'
  }

  const badgeText = (clip) => {
    if (clip.correct) return 'OK'
    if (clip.status === 'stage1_fail') return 'S1 FAIL'
    if (clip.status === 'stage2_fail') return 'S2 FAIL'
    if (clip.status === 'stage3_fail') return 'S3 FAIL'
    if (clip.row_err !== undefined && clip.row_err !== null) {
      return `r${clip.row_err > 0 ? '+' : ''}${clip.row_err} c${clip.col_err > 0 ? '+' : ''}${clip.col_err}`
    }
    return 'MISS'
  }

  // Keyboard nav
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return
      if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault()
        const idx = filteredClips.findIndex(c => c.clip_name === selectedClip?.clip_name)
        if (idx < filteredClips.length - 1) setSelectedClip(filteredClips[idx + 1])
      }
      if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault()
        const idx = filteredClips.findIndex(c => c.clip_name === selectedClip?.clip_name)
        if (idx > 0) setSelectedClip(filteredClips[idx - 1])
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  return (
    <div className="page-content">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>End-to-End Runs</h2>
          {runs.length > 0 ? (
            <select
              className="run-select"
              value={selectedRun || ''}
              onChange={(e) => setSelectedRun(e.target.value)}
            >
              {runs.map(r => (
                <option key={r.name} value={r.name}>
                  {r.name} ({r.summary.exact_pct}%)
                </option>
              ))}
            </select>
          ) : (
            <div className="no-runs">No E2E runs yet.</div>
          )}

          {runData && (
            <div className="run-summary">
              {hasFolds && (
                <div style={{ fontSize: 10, color: '#60a5fa', background: '#172554', padding: '3px 8px', borderRadius: 4, marginBottom: 6, textAlign: 'center', fontWeight: 600 }}>
                  5-Fold Stratified CV
                </div>
              )}
              <div className="summary-row">
                <span>Exact match</span>
                <span className="stat-ok">
                  {summary.exact_match}/{summary.total_clips} ({summary.exact_pct}%)
                </span>
              </div>
              <div className="summary-row">
                <span>Row OK</span>
                <span>{summary.row_ok}/{summary.total_clips} ({summary.row_ok_pct}%)</span>
              </div>
              <div className="summary-row">
                <span>Col OK</span>
                <span>{summary.col_ok}/{summary.total_clips} ({summary.col_ok_pct}%)</span>
              </div>
              <div className="summary-row">
                <span>Within 1</span>
                <span className="stat-close">{summary.within_1}/{summary.total_clips} ({summary.within_1_pct}%)</span>
              </div>
              <div className="summary-row">
                <span>Within 2</span>
                <span>{summary.within_2}/{summary.total_clips} ({summary.within_2_pct}%)</span>
              </div>
              {summary.type_correct !== undefined && (
                <div className="summary-row">
                  <span>Type correct</span>
                  <span style={{ color: summary.type_correct_pct >= 80 ? '#4ade80' : summary.type_correct_pct >= 60 ? '#fbbf24' : '#f87171' }}>
                    {summary.type_correct}/{summary.total_clips} ({summary.type_correct_pct}%)
                  </span>
                </div>
              )}
              <div className="summary-divider" />
              <div className="summary-row">
                <span>S1 fails</span>
                <span style={{ color: summary.stage1_fails > 0 ? '#f87171' : '#555' }}>{summary.stage1_fails}</span>
              </div>
              <div className="summary-row">
                <span>S2 fails</span>
                <span style={{ color: summary.stage2_fails > 0 ? '#f87171' : '#555' }}>{summary.stage2_fails}</span>
              </div>
              <div className="summary-row">
                <span>S3 fails</span>
                <span style={{ color: summary.stage3_fails > 0 ? '#f87171' : '#555' }}>{summary.stage3_fails}</span>
              </div>

              {Object.keys(plateBreakdown).length > 0 && (
                <>
                  <div className="summary-divider" />
                  {Object.entries(plateBreakdown).sort(([a], [b]) => {
                    const na = parseInt(a.replace(/\D/g, '')) || 0
                    const nb = parseInt(b.replace(/\D/g, '')) || 0
                    return na - nb
                  }).map(([plate, stats]) => (
                    <div key={plate} className="summary-row">
                      <span>{plate}</span>
                      <span style={{ color: stats.accuracy_pct >= 50 ? '#4ade80' : stats.accuracy_pct >= 30 ? '#fbbf24' : '#f87171' }}>
                        {stats.correct}/{stats.total} ({stats.accuracy_pct}%)
                      </span>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}

          <div className="tab-bar" style={{ marginTop: 10 }}>
            {['all', 'correct', 'wrong', 'fail'].map(f => (
              <button
                key={f}
                className={`tab-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f}<span className="count">{countByFilter(f)}</span>
              </button>
            ))}
          </div>

          {hasFolds && (
            <div className="tab-bar" style={{ marginTop: 6 }}>
              <button
                className={`tab-btn ${groupBy === 'plate' ? 'active' : ''}`}
                onClick={() => setGroupBy('plate')}
                style={{ fontSize: 11 }}
              >
                By Plate
              </button>
              <button
                className={`tab-btn ${groupBy === 'fold' ? 'active' : ''}`}
                onClick={() => setGroupBy('fold')}
                style={{ fontSize: 11 }}
              >
                By Fold
              </button>
            </div>
          )}
        </div>

        <div className="clip-list">
          {groupKeys.map(groupKey => {
            const groupClips = groupedClips[groupKey]
            const ps = groupBy === 'plate' ? plateBreakdown[groupKey] : null
            // Compute fold accuracy if grouping by fold
            const foldStats = groupBy === 'fold' ? (() => {
              const correct = groupClips.filter(c => c.correct).length
              const total = groupClips.length
              return { correct, total, accuracy_pct: total ? Math.round(1000 * correct / total) / 10 : 0 }
            })() : null
            const stats = ps || foldStats
            return (
              <div key={groupKey}>
                <div className="plate-header">
                  {groupKey} ({groupClips.length})
                  {stats && (
                    <span style={{
                      fontSize: 10,
                      color: stats.accuracy_pct >= 50 ? '#4ade80' : stats.accuracy_pct >= 30 ? '#fbbf24' : '#f87171',
                    }}>
                      {stats.accuracy_pct}%
                    </span>
                  )}
                </div>
                {groupClips.map(clip => {
                  const short = clip.clip_name.replace(/^Plate_\d+_/, '')
                  const isActive = selectedClip?.clip_name === clip.clip_name
                  return (
                    <div
                      key={clip.clip_name}
                      className={`clip-item ${isActive ? 'active' : ''}`}
                      onClick={() => setSelectedClip(clip)}
                    >
                      <span className="name">
                        {short}
                        {hasFolds && groupBy === 'plate' && (
                          <span style={{ fontSize: 9, color: '#555', marginLeft: 4 }}>f{clip.fold}</span>
                        )}
                      </span>
                      <span className="clip-badges">
                        {clip.pred_well && (
                          <span style={{ fontSize: 11, color: '#888', marginRight: 4 }}>
                            {clip.pred_well}
                          </span>
                        )}
                        <span className={`badge ${badgeClass(clip)}`}>
                          {badgeText(clip)}
                        </span>
                      </span>
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>

        <div className="keyboard-hint">
          <kbd>j</kbd><kbd>k</kbd> or <kbd>&uarr;</kbd><kbd>&darr;</kbd> navigate clips
        </div>
      </div>

      <div className="main">
        {selectedClip ? (
          <div className="e2e-stages-view">
            {/* Result panel */}
            <div className="well-result-panel">
              <div className="well-result-row">
                <span className="well-label">Clip:</span>
                <span className="well-value">{selectedClip.clip_name}</span>
                {selectedClip.fold !== undefined && (
                  <span style={{ fontSize: 10, color: '#60a5fa', background: '#172554', padding: '1px 6px', borderRadius: 3, fontWeight: 600, marginLeft: 4 }}>
                    Fold {selectedClip.fold}
                  </span>
                )}
              </div>
              <div className="well-result-row">
                <span className="well-label">Predicted:</span>
                <span className="well-value">{selectedClip.pred_well || '—'}</span>
              </div>
              <div className="well-result-row">
                <span className="well-label">Ground Truth:</span>
                <span className="well-value">{(selectedClip.gt_wells || []).join(', ')}</span>
              </div>
              <div className="well-result-row">
                {selectedClip.status === 'ok' ? (
                  <span className={`well-verdict ${selectedClip.correct ? 'correct' : 'incorrect'}`}>
                    {selectedClip.correct ? 'CORRECT' : 'INCORRECT'}
                  </span>
                ) : (
                  <span className="well-verdict incorrect">{selectedClip.status?.toUpperCase()}</span>
                )}
                {selectedClip.pipette_type && (
                  <span className="pipette-type-tag" style={{ marginLeft: 8 }}>
                    {selectedClip.pipette_type}
                    {selectedClip.gt_pipette_type && selectedClip.pipette_type !== selectedClip.gt_pipette_type && (
                      <span style={{ color: '#f87171' }}> (gt: {selectedClip.gt_pipette_type})</span>
                    )}
                  </span>
                )}
                {selectedClip.row_err !== undefined && selectedClip.row_err !== null && !selectedClip.correct && (
                  <span style={{ fontSize: 12, color: '#888', marginLeft: 12 }}>
                    row err: {selectedClip.row_err}, col err: {selectedClip.col_err}
                  </span>
                )}
              </div>
            </div>

            {/* Stage overlays */}
            <div className="e2e-stages-scroll">
              {/* Summary */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">Summary</div>
                <img
                  src={`/e2e_images/${selectedRun}/${selectedClip.overlays?.summary}`}
                  alt="Summary"
                  className="e2e-stage-img"
                />
              </div>

              {/* Stage 1: Commit Frame */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">
                  <span className="e2e-stage-num">1</span>
                  Commit Frame Detection
                  <span className="e2e-stage-detail">
                    Frame {selectedClip.commit_frame_idx ?? '—'}
                  </span>
                  <span className={`e2e-stage-status ${selectedClip.status === 'stage1_fail' ? 'fail' : 'ok'}`}>
                    {selectedClip.status === 'stage1_fail' ? 'FAIL' : 'OK'}
                  </span>
                </div>
                <img
                  src={`/e2e_images/${selectedRun}/${selectedClip.overlays?.commit}`}
                  alt="Stage 1"
                  className="e2e-stage-img"
                />
              </div>

              {/* Stage 2: Plate Corners */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">
                  <span className="e2e-stage-num">2</span>
                  Plate Corner Detection
                  <span className="e2e-stage-detail">
                    {selectedClip.n_corners_detected}/4 corners
                  </span>
                  <span className={`e2e-stage-status ${selectedClip.status === 'stage2_fail' ? 'fail' : 'ok'}`}>
                    {selectedClip.status === 'stage2_fail' ? 'FAIL' : 'OK'}
                  </span>
                </div>
                <img
                  src={`/e2e_images/${selectedRun}/${selectedClip.overlays?.corners}`}
                  alt="Stage 2"
                  className="e2e-stage-img"
                />
              </div>

              {/* Stage 3: Tip Detection */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">
                  <span className="e2e-stage-num">3</span>
                  Tip Detection (HeatNet)
                  <span className="e2e-stage-detail">
                    Pred: {selectedClip.pred_well || '—'}
                  </span>
                  <span className={`e2e-stage-status ${selectedClip.status === 'stage3_fail' ? 'fail' : 'ok'}`}>
                    {selectedClip.status === 'stage3_fail' ? 'FAIL' : 'OK'}
                  </span>
                </div>
                <img
                  src={`/e2e_images/${selectedRun}/${selectedClip.overlays?.warped}`}
                  alt="Stage 3"
                  className="e2e-stage-img"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="exp-image-area">
            <div className="no-prediction">Select a clip to view results</div>
          </div>
        )}
      </div>
    </div>
  )
}
