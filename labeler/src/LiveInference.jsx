import { useState, useRef } from 'react'

export default function LiveInference() {
  const [clipPairs, setClipPairs] = useState([{ fpv: null, topview: null }])
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [results, setResults] = useState(null)
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [error, setError] = useState(null)

  const handleFileChange = (idx, type, file) => {
    setClipPairs(prev => {
      const next = [...prev]
      next[idx] = { ...next[idx], [type]: file }
      return next
    })
  }

  const addPair = () => {
    setClipPairs(prev => [...prev, { fpv: null, topview: null }])
  }

  const removePair = (idx) => {
    if (clipPairs.length <= 1) return
    setClipPairs(prev => prev.filter((_, i) => i !== idx))
  }

  const canRun = clipPairs.some(p => p.fpv && p.topview) && !isRunning

  const runInference = async () => {
    setIsRunning(true)
    setError(null)
    setResults(null)
    setProgress('Uploading videos and running pipeline...')

    try {
      const formData = new FormData()
      const validPairs = clipPairs.filter(p => p.fpv && p.topview)

      if (validPairs.length === 1) {
        formData.append('fpv_video', validPairs[0].fpv)
        formData.append('topview_video', validPairs[0].topview)
      } else {
        validPairs.forEach((pair, i) => {
          formData.append(`fpv_video_${i}`, pair.fpv)
          formData.append(`topview_video_${i}`, pair.topview)
        })
      }

      const res = await fetch('/api/inference', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        let msg = `Server error (${res.status})`
        try {
          const err = await res.json()
          msg = err.error || msg
        } catch {}
        throw new Error(msg)
      }

      const data = await res.json()
      setSessionId(data.session_id)
      setResults(data.results)
      setSelectedIdx(0)
      setProgress('')
    } catch (e) {
      setError(e.message)
      setProgress('')
    } finally {
      setIsRunning(false)
    }
  }

  const selectedResult = results?.[selectedIdx]

  const stageStatus = (result, stage) => {
    if (!result) return 'pending'
    if (result.status === 'ok') return 'ok'
    if (result.status === `${stage}_fail`) return 'fail'
    // Check if this stage passed (it's a later stage that failed)
    const stageOrder = ['stage1', 'stage2', 'stage3']
    const failIdx = stageOrder.indexOf(result.status.replace('_fail', ''))
    const thisIdx = stageOrder.indexOf(stage)
    if (failIdx >= 0 && thisIdx < failIdx) return 'ok'
    if (failIdx >= 0 && thisIdx === failIdx) return 'fail'
    if (failIdx >= 0 && thisIdx > failIdx) return 'skipped'
    return 'ok'
  }

  return (
    <div className="page-content">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>Live Inference</h2>

          {/* Upload area */}
          <div className="li-upload-area">
            {clipPairs.map((pair, idx) => (
              <div key={idx} className="li-pair">
                <div className="li-pair-header">
                  <span className="li-pair-label">Clip {idx + 1}</span>
                  {clipPairs.length > 1 && (
                    <button className="li-remove-btn" onClick={() => removePair(idx)}>x</button>
                  )}
                </div>
                <div className="li-file-row">
                  <label className="li-file-label">
                    <span className="li-file-tag fpv">FPV</span>
                    <input
                      type="file"
                      accept="video/mp4,video/*"
                      onChange={(e) => handleFileChange(idx, 'fpv', e.target.files[0])}
                      className="li-file-input"
                    />
                    <span className="li-file-name">
                      {pair.fpv ? pair.fpv.name : 'Choose file...'}
                    </span>
                  </label>
                </div>
                <div className="li-file-row">
                  <label className="li-file-label">
                    <span className="li-file-tag topview">Top</span>
                    <input
                      type="file"
                      accept="video/mp4,video/*"
                      onChange={(e) => handleFileChange(idx, 'topview', e.target.files[0])}
                      className="li-file-input"
                    />
                    <span className="li-file-name">
                      {pair.topview ? pair.topview.name : 'Choose file...'}
                    </span>
                  </label>
                </div>
              </div>
            ))}

            <button className="li-add-btn" onClick={addPair}>
              + Add another clip
            </button>

            <button
              className="li-run-btn"
              disabled={!canRun}
              onClick={runInference}
            >
              {isRunning ? 'Running pipeline...' : 'Run Inference'}
            </button>

            {progress && <div className="li-progress">{progress}</div>}
            {error && <div className="li-error">{error}</div>}
          </div>
        </div>

        {/* Results list (for batch) */}
        {results && results.length > 1 && (
          <div className="clip-list">
            {results.map((r, i) => {
              const name = r.clip_id_FPV?.replace('_FPV', '') || `Clip ${i + 1}`
              const short = name.replace(/^Plate_\d+_/, '')
              return (
                <div
                  key={i}
                  className={`clip-item ${selectedIdx === i ? 'active' : ''}`}
                  onClick={() => setSelectedIdx(i)}
                >
                  <span className="name">{short}</span>
                  <span className={`badge ${r.status === 'ok' ? 'ok' : 'fail'}`}>
                    {r.status === 'ok' ? 'OK' : r.status?.toUpperCase()}
                  </span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="main">
        {selectedResult ? (
          <div className="e2e-stages-view">
            {/* Prediction result */}
            <div className="well-result-panel">
              <div className="well-result-row">
                <span className="well-label">Clip:</span>
                <span className="well-value">
                  {selectedResult.clip_id_FPV?.replace('_FPV', '') || '—'}
                </span>
              </div>
              <div className="well-result-row">
                <span className="well-label">Prediction:</span>
                <span className="well-value" style={{ fontFamily: 'monospace' }}>
                  {selectedResult.wells_prediction?.map(
                    w => `${w.well_row}${w.well_column}`
                  ).join(', ') || '—'}
                </span>
              </div>
              <div className="well-result-row">
                <span className="well-label">Pipette type:</span>
                <span className="well-value">{selectedResult.pipette_type || '—'}</span>
              </div>
              <div className="well-result-row">
                <span className={`well-verdict ${selectedResult.status === 'ok' ? 'correct' : 'incorrect'}`}>
                  {selectedResult.status === 'ok' ? 'PIPELINE OK' : selectedResult.status?.toUpperCase()}
                </span>
              </div>
            </div>

            {/* Stage overlays */}
            <div className="e2e-stages-scroll">
              {/* Summary */}
              {selectedResult.overlays?.summary && (
                <div className="e2e-stage-panel">
                  <div className="e2e-stage-label">Summary</div>
                  <img
                    src={`/inference_images/${sessionId}/${selectedResult.overlays.summary}`}
                    alt="Summary"
                    className="e2e-stage-img"
                  />
                </div>
              )}

              {/* Stage 1 */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">
                  <span className="e2e-stage-num">1</span>
                  Commit Frame Detection
                  <span className="e2e-stage-detail">
                    Frame {selectedResult.stages?.commit_frame?.frame_idx ?? '—'}
                  </span>
                  <span className={`e2e-stage-status ${stageStatus(selectedResult, 'stage1') === 'ok' ? 'ok' : 'fail'}`}>
                    {stageStatus(selectedResult, 'stage1').toUpperCase()}
                  </span>
                </div>
                {selectedResult.overlays?.commit && (
                  <img
                    src={`/inference_images/${sessionId}/${selectedResult.overlays.commit}`}
                    alt="Stage 1"
                    className="e2e-stage-img"
                  />
                )}
              </div>

              {/* Stage 2 */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">
                  <span className="e2e-stage-num">2</span>
                  Plate Corner Detection
                  <span className="e2e-stage-detail">
                    {selectedResult.stages?.plate_detection?.n_corners ?? '?'}/4 corners
                  </span>
                  <span className={`e2e-stage-status ${stageStatus(selectedResult, 'stage2') === 'ok' ? 'ok' : stageStatus(selectedResult, 'stage2') === 'skipped' ? 'skipped' : 'fail'}`}>
                    {stageStatus(selectedResult, 'stage2').toUpperCase()}
                  </span>
                </div>
                {selectedResult.overlays?.corners && (
                  <img
                    src={`/inference_images/${sessionId}/${selectedResult.overlays.corners}`}
                    alt="Stage 2"
                    className="e2e-stage-img"
                  />
                )}
              </div>

              {/* Stage 3 */}
              <div className="e2e-stage-panel">
                <div className="e2e-stage-label">
                  <span className="e2e-stage-num">3</span>
                  Tip Detection (HeatNet v2)
                  <span className="e2e-stage-detail">
                    Pred: {selectedResult.wells_prediction?.[0]
                      ? `${selectedResult.wells_prediction[0].well_row}${selectedResult.wells_prediction[0].well_column}`
                      : '—'}
                    {selectedResult.stages?.tip_detection?.pred_type && (
                      <> · Type: {selectedResult.stages.tip_detection.pred_type}</>
                    )}
                  </span>
                  <span className={`e2e-stage-status ${stageStatus(selectedResult, 'stage3') === 'ok' ? 'ok' : stageStatus(selectedResult, 'stage3') === 'skipped' ? 'skipped' : 'fail'}`}>
                    {stageStatus(selectedResult, 'stage3').toUpperCase()}
                  </span>
                </div>
                {selectedResult.overlays?.warped && (
                  <img
                    src={`/inference_images/${sessionId}/${selectedResult.overlays.warped}`}
                    alt="Stage 3"
                    className="e2e-stage-img"
                  />
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="exp-image-area">
            <div className="no-prediction">
              {isRunning ? 'Running pipeline...' : 'Upload videos and run inference'}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
