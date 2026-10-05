import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

function RouteBadge({ route, method }) {
  return (
    <span className={`route-badge route-${route}`}>
      {route} <span className="route-method">· {method}</span>
    </span>
  )
}

function App() {
  const [docText, setDocText] = useState('')
  const [docId, setDocId] = useState('')
  const [ingestResult, setIngestResult] = useState(null)
  const [ingestLoading, setIngestLoading] = useState(false)
  const [ingestError, setIngestError] = useState(null)

  const [question, setQuestion] = useState('Who does John Smith report to?')
  const [queryResult, setQueryResult] = useState(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState(null)

  const loadSample = async () => {
    setIngestError(null)
    try {
      const res = await fetch(`${API_URL}/api/sample-document`)
      const data = await res.json()
      setDocText(data.text)
      setDocId('q3_report')
    } catch (err) {
      setIngestError('Could not reach the backend. Is it running?')
    }
  }

  const ingest = async () => {
    setIngestLoading(true)
    setIngestError(null)
    setIngestResult(null)
    try {
      const res = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: docText, doc_id: docId || undefined }),
      })
      if (!res.ok) throw new Error(`Server responded with ${res.status}`)
      setIngestResult(await res.json())
    } catch (err) {
      setIngestError(err.message || 'Something went wrong reaching the backend.')
    } finally {
      setIngestLoading(false)
    }
  }

  const askQuestion = async () => {
    setQueryLoading(true)
    setQueryError(null)
    setQueryResult(null)
    try {
      const res = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!res.ok) throw new Error(`Server responded with ${res.status}`)
      setQueryResult(await res.json())
    } catch (err) {
      setQueryError(err.message || 'Something went wrong reaching the backend.')
    } finally {
      setQueryLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🕸️ Enterprise Knowledge-Graph RAG</h1>
        <p>Hybrid vector + graph retrieval, with semantic routing between them</p>
      </header>

      <section className="panel">
        <div className="panel-header">
          <h2>1. Ingest a document</h2>
          <button className="btn-secondary" onClick={loadSample}>Load sample report</button>
        </div>

        <textarea
          value={docText}
          onChange={(e) => setDocText(e.target.value)}
          placeholder="Paste messy document text here (or load the sample report)…"
          rows={8}
        />

        <div className="actions">
          <input
            className="doc-id-input"
            value={docId}
            onChange={(e) => setDocId(e.target.value)}
            placeholder="doc_id (optional)"
          />
          <button className="btn-primary" onClick={ingest} disabled={ingestLoading || !docText.trim()}>
            {ingestLoading ? 'Ingesting…' : 'Ingest document'}
          </button>
        </div>

        {ingestError && <div className="error-banner">{ingestError}</div>}

        {ingestResult && (
          <div className="ingest-summary">
            <span>doc_id: <strong>{ingestResult.doc_id}</strong></span>
            <span>chunks: <strong>{ingestResult.chunks_added}</strong></span>
            <span>triples: <strong>{ingestResult.triples_added}</strong></span>
            <span>extraction: <strong>{ingestResult.extraction_method}</strong></span>
            <span>graph backend: <strong>{ingestResult.graph_backend}</strong></span>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>2. Ask a question</h2>
        </div>

        <div className="query-row">
          <input
            className="question-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. Who does John Smith report to?"
            onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
          />
          <button className="btn-primary" onClick={askQuestion} disabled={queryLoading || !question.trim()}>
            {queryLoading ? 'Thinking…' : 'Ask'}
          </button>
        </div>

        {queryError && <div className="error-banner">{queryError}</div>}

        {queryResult && (
          <div className="result">
            <div className="result-route">
              <RouteBadge route={queryResult.route} method={queryResult.route_method} />
              <span className="route-reason">{queryResult.route_reason}</span>
            </div>

            <div className="answer-box">
              <span className="answer-label">Answer ({queryResult.answer_method})</span>
              <p>{queryResult.answer}</p>
            </div>

            {queryResult.graph_triples.length > 0 && (
              <div className="context-block">
                <h3>Graph context ({queryResult.graph_triples.length} relationships)</h3>
                <ul className="triple-list">
                  {queryResult.graph_triples.map((t, i) => (
                    <li key={i}>
                      <span className="entity">{t.subject}</span>
                      <span className="relation">{t.relation}</span>
                      <span className="entity">{t.object}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {queryResult.vector_sources.length > 0 && (
              <div className="context-block">
                <h3>Vector context ({queryResult.vector_sources.length} chunks)</h3>
                {queryResult.vector_sources.map((s, i) => (
                  <div className="source-chunk" key={i}>
                    {s.section && <span className="section-tag">{s.section}</span>}
                    <span className="score-tag">score {s.score}</span>
                    <p>{s.text}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  )
}

export default App