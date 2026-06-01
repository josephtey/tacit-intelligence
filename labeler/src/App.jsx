import { useState } from 'react'
import './App.css'
import EndToEndRuns from './EndToEndRuns'
import LiveInference from './LiveInference'

function App() {
  const [page, setPage] = useState('e2e')

  return (
    <div className="app">
      <div className="page-nav">
        <button
          className={`page-nav-btn ${page === 'e2e' ? 'active' : ''}`}
          onClick={() => setPage('e2e')}
        >
          End-to-End Runs
        </button>
        <button
          className={`page-nav-btn ${page === 'inference' ? 'active' : ''}`}
          onClick={() => setPage('inference')}
        >
          Live Inference
        </button>
      </div>
      {page === 'e2e' && <EndToEndRuns />}
      {page === 'inference' && <LiveInference />}
    </div>
  )
}

export default App
