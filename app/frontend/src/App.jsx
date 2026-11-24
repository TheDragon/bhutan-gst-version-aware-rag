import { useEffect, useMemo, useState } from 'react'
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Clock3,
  ExternalLink,
  FileCheck2,
  FileSearch,
  Gauge,
  History,
  Layers3,
  Library,
  Menu,
  Search,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import { api } from './api'

const NAV = [
  { id: 'overview', label: 'Overview' },
  { id: 'research', label: 'Ask & inspect' },
  { id: 'compare', label: 'Compare systems' },
  { id: 'timeline', label: 'Version timeline' },
  { id: 'library', label: 'Corpus library' },
]

const SYSTEMS = {
  standard: { label: 'Standard RAG', short: 'Standard', tone: 'slate' },
  date_filtered: { label: 'Date-filtered RAG', short: 'Date-filtered', tone: 'blue' },
  version_aware: { label: 'Version-aware RAG', short: 'Version-aware', tone: 'gold' },
}

const EXAMPLES = [
  'As of 1 January 2026, how was Section 40 amended?',
  'What does Section 11 say about taxable activity?',
  'Which GST provisions came into force in 2026?',
]

const today = new Date().toISOString().slice(0, 10)

function cx(...values) {
  return values.filter(Boolean).join(' ')
}

function App() {
  const [page, setPage] = useState('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const [stats, setStats] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    Promise.all([api.stats(), api.health()])
      .then(([statsData, healthData]) => {
        setStats(statsData)
        setHealth(healthData)
      })
      .catch(() => {})
  }, [])

  const navigate = (next) => {
    setPage(next)
    setMenuOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="app-shell">
      <Header page={page} navigate={navigate} menuOpen={menuOpen} setMenuOpen={setMenuOpen} />
      <main>
        {page === 'overview' && <Overview stats={stats} health={health} navigate={navigate} />}
        {page === 'research' && <ResearchWorkspace />}
        {page === 'compare' && <CompareWorkspace />}
        {page === 'timeline' && <TimelineExplorer />}
        {page === 'library' && <CorpusLibrary />}
      </main>
      <Footer health={health} />
    </div>
  )
}

function Header({ page, navigate, menuOpen, setMenuOpen }) {
  return (
    <header className="site-header">
      <button className="brand" onClick={() => navigate('overview')} aria-label="GST Lens home">
        <img src="/gst-lens-logo.png" alt="" />
        <span><strong>GST Lens</strong><small>Bhutan legislation research</small></span>
      </button>
      <nav className={cx('main-nav', menuOpen && 'open')} aria-label="Main navigation">
        {NAV.map((item) => (
          <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => navigate(item.id)}>
            {item.label}
          </button>
        ))}
      </nav>
      <div className="header-actions">
        <span className="prototype-pill"><span /> Research prototype</span>
        <button className="menu-button" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle navigation">
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>
    </header>
  )
}

function Overview({ stats, health, navigate }) {
  const statCards = [
    { value: stats?.documents ?? '10', label: 'Official documents', icon: FileCheck2 },
    { value: stats?.chunks?.toLocaleString() ?? '1,581', label: 'Searchable passages', icon: Layers3 },
    { value: stats?.version_events ?? '62', label: 'Version relationships', icon: History },
    { value: '69.6%', label: 'Version accuracy', icon: Gauge },
  ]

  return (
    <>
      <section className="hero section-wrap">
        <div className="hero-copy">
          <span className="eyebrow"><ShieldCheck size={15} /> Evidence first · Date aware</span>
          <h1>Find the GST rule that applied <em>then</em>—not only the one that applies now.</h1>
          <p>
            A version-aware research workspace for Bhutan GST legislation. Ask a dated question,
            inspect the underlying passage, trace amendments, and compare retrieval methods.
          </p>
          <div className="hero-actions">
            <button className="primary-button" onClick={() => navigate('research')}>
              Start researching <ArrowRight size={17} />
            </button>
            <button className="secondary-button" onClick={() => navigate('timeline')}>
              Explore versions
            </button>
          </div>
          <div className="model-line">
            <span><Check size={14} /> Fine-tuned MiniLM</span>
            <span><Check size={14} /> BM25 + FAISS</span>
            <span><Check size={14} /> {health?.generator_model?.replace('google/', '') || 'FLAN-T5-base'}</span>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="shape shape-a" />
          <div className="shape shape-b" />
          <div className="shape shape-c" />
          <div className="research-card card-main">
            <span className="mini-label">Question with date</span>
            <p>How was Section 40 amended?</p>
            <div className="date-chip"><CalendarDays size={14} /> 01 Jan 2026</div>
          </div>
          <div className="research-card card-source">
            <span className="source-index">01</span>
            <div><strong>Amendment Act 2025</strong><small>Section 12 · page 4</small></div>
            <FileCheck2 size={20} />
          </div>
          <div className="research-card card-version">
            <History size={18} />
            <div><strong>Applicable version</strong><small>Effective 01 Jan 2026</small></div>
          </div>
        </div>
      </section>

      <section className="stats-band section-wrap" aria-label="Project statistics">
        {statCards.map(({ value, label, icon: Icon }) => (
          <article className="stat-card" key={label}>
            <Icon size={19} />
            <strong>{value}</strong>
            <span>{label}</span>
          </article>
        ))}
      </section>

      <section className="feature-section section-wrap">
        <div className="section-heading split-heading">
          <div>
            <span className="eyebrow">A research workspace, not a chatbot</span>
            <h2>Move from question to verified source.</h2>
          </div>
          <p>Every result stays connected to its document, section, page, effective period, and official PDF.</p>
        </div>
        <div className="feature-grid">
          <FeatureCard icon={FileSearch} accent="blue" number="01" title="Ask & inspect" text="Ask naturally, select a target date, and inspect how lexical and semantic retrieval found each passage." onClick={() => navigate('research')} />
          <FeatureCard icon={BarChart3} accent="pink" number="02" title="Compare systems" text="Run the same question through standard, date-filtered, and version-aware retrieval side by side." onClick={() => navigate('compare')} />
          <FeatureCard icon={History} accent="gold" number="03" title="Trace versions" text="Follow base text, amendments, effective dates, and source-linked version evidence for a section." onClick={() => navigate('timeline')} />
          <FeatureCard icon={Library} accent="green" number="04" title="Browse the corpus" text="Review all indexed Acts, amendments, rules, notices, extraction status, and source links." onClick={() => navigate('library')} />
        </div>
      </section>

      <section className="how-section section-wrap">
        <div className="how-copy">
          <span className="eyebrow">How it works</span>
          <h2>One query. Four transparent steps.</h2>
          <p>Retrieval and generation are kept visible so a reviewer can understand where an answer came from.</p>
        </div>
        <ol className="process-row">
          <li><span>1</span><strong>Ask</strong><small>Question + target date</small></li>
          <li><span>2</span><strong>Retrieve</strong><small>BM25 + FAISS</small></li>
          <li><span>3</span><strong>Select version</strong><small>Dates + amendments</small></li>
          <li><span>4</span><strong>Answer & cite</strong><small>Evidence-bound output</small></li>
        </ol>
      </section>
    </>
  )
}

function FeatureCard({ icon: Icon, accent, number, title, text, onClick }) {
  return (
    <button className={cx('feature-card', accent)} onClick={onClick}>
      <span className="feature-top"><span className="feature-icon"><Icon size={23} /></span><small>{number}</small></span>
      <strong>{title}</strong>
      <p>{text}</p>
      <span className="card-link">Open workspace <ChevronRight size={15} /></span>
    </button>
  )
}

function ResearchWorkspace() {
  const [question, setQuestion] = useState(EXAMPLES[0])
  const [targetDate, setTargetDate] = useState('2026-01-01')
  const [system, setSystem] = useState('version_aware')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async (event) => {
    event?.preventDefault()
    setLoading(true)
    setError('')
    try {
      setResult(await api.answer({ question, target_date: targetDate || null, system, top_k: 5 }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageFrame eyebrow="Ask & inspect" title="Research a GST question" description="Choose the legal date, inspect the answer, then audit every retrieved passage and score.">
      <div className="workspace-grid">
        <aside className="query-panel">
          <form onSubmit={run}>
            <label htmlFor="question">Your question</label>
            <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} rows="6" />
            <div className="field-row">
              <div className="field-group">
                <label htmlFor="target-date">Target date</label>
                <div className="input-with-icon"><CalendarDays size={17} /><input id="target-date" type="date" max={today} value={targetDate} onChange={(e) => setTargetDate(e.target.value)} /></div>
              </div>
              <div className="field-group">
                <label htmlFor="system">Retrieval mode</label>
                <div className="select-wrap"><select id="system" value={system} onChange={(e) => setSystem(e.target.value)}>{Object.entries(SYSTEMS).map(([key, value]) => <option value={key} key={key}>{value.label}</option>)}</select><ChevronDown size={16} /></div>
              </div>
            </div>
            <button className="primary-button full" disabled={loading || question.trim().length < 3}>
              {loading ? <><span className="spinner" /> Loading models & researching…</> : <><Sparkles size={17} /> Research question</>}
            </button>
          </form>
          <div className="example-block">
            <span>Try an example</span>
            {EXAMPLES.map((example) => <button key={example} onClick={() => setQuestion(example)}>{example}<ChevronRight size={14} /></button>)}
          </div>
          <div className="help-note"><CircleAlert size={16} /><p>The first generated answer may take longer while FLAN-T5-base loads into memory.</p></div>
        </aside>

        <section className="result-panel">
          {!result && !loading && !error && <EmptyState icon={FileSearch} title="Your evidence will appear here" text="Ask a question to see the generated answer, applicable date, citations, and retrieval evidence." />}
          {loading && <LoadingState />}
          {error && <ErrorState message={error} />}
          {result && !loading && (
            <>
              <div className="answer-card">
                <div className="answer-meta">
                  <span className="status-dot"><span /> Evidence-bound answer</span>
                  <SystemBadge system={result.system} />
                </div>
                <h2>{result.abstained ? 'Insufficient evidence' : 'Answer'}</h2>
                <p className="answer-text">{result.answer}</p>
                <div className="answer-footer">
                  <span><CalendarDays size={15} /> Applicable on {formatDate(result.target_date)}</span>
                  <span><Sparkles size={15} /> {result.generator?.replace('google/', '')}</span>
                  {result.answer_strategy && (
                    <span><ShieldCheck size={15} /> {result.answer_strategy.replaceAll('_', ' ')}</span>
                  )}
                </div>
              </div>
              <div className="notice-banner"><ShieldCheck size={18} /><span><strong>Verify before relying.</strong> {result.notice}</span></div>
              <EvidenceList evidence={result.evidence} />
            </>
          )}
        </section>
      </div>
    </PageFrame>
  )
}

function CompareWorkspace() {
  const [question, setQuestion] = useState(EXAMPLES[0])
  const [targetDate, setTargetDate] = useState('2026-01-01')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      setResult(await api.compare({ question, target_date: targetDate || null, top_k: 3, generate_answers: false }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageFrame eyebrow="Controlled comparison" title="See what version awareness changes" description="Compare the top evidence returned by all three systems using the same question and date.">
      <form className="compare-form" onSubmit={run}>
        <div><label htmlFor="compare-question">Question</label><input id="compare-question" value={question} onChange={(e) => setQuestion(e.target.value)} /></div>
        <div><label htmlFor="compare-date">Target date</label><input id="compare-date" type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} /></div>
        <button className="primary-button" disabled={loading}>{loading ? <span className="spinner" /> : <BarChart3 size={17} />} Compare</button>
      </form>
      {error && <ErrorState message={error} />}
      {!result && !loading && <div className="comparison-intro"><BarChart3 size={26} /><div><strong>Why compare?</strong><p>Standard retrieval can return a relevant but outdated passage. Date filtering removes inactive documents. Version-aware retrieval also follows amendment relationships and prioritises the applicable legal version.</p></div></div>}
      {loading && <LoadingState compact />}
      {result && !loading && (
        <div className="comparison-grid">
          {result.comparisons.map((column) => (
            <article className={cx('comparison-column', column.system)} key={column.system}>
              <div className="comparison-head"><SystemBadge system={column.system} /><small>{column.evidence.length} passages</small></div>
              {column.evidence.map((item, index) => <CompactEvidence item={item} index={index} key={item.chunk_id} />)}
            </article>
          ))}
        </div>
      )}
    </PageFrame>
  )
}

function TimelineExplorer() {
  const [section, setSection] = useState('40')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async (event) => {
    event?.preventDefault()
    setLoading(true)
    setError('')
    try {
      setResult(await api.timeline(section))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { run() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <PageFrame eyebrow="Version explorer" title="Trace a section through time" description="Review base provisions and amendment evidence connected to a GST section.">
      <div className="timeline-layout">
        <aside className="timeline-controls">
          <form onSubmit={run}>
            <label htmlFor="section-number">GST section</label>
            <div className="search-input"><Search size={17} /><input id="section-number" value={section} onChange={(e) => setSection(e.target.value)} placeholder="e.g. 40" /><button aria-label="Search section"><ArrowRight size={17} /></button></div>
          </form>
          <div className="popular-sections"><span>Quick sections</span><div>{['3', '5', '11', '40', '70'].map((item) => <button className={section === item ? 'active' : ''} key={item} onClick={() => setSection(item)}>{item}</button>)}</div></div>
          <div className="help-note"><History size={16} /><p>Entries include both the section text and amendment passages that explicitly affect it.</p></div>
        </aside>
        <section className="timeline-panel">
          {loading && <LoadingState compact />}
          {error && <ErrorState message={error} />}
          {result && !loading && (
            <>
              <div className="timeline-summary"><div><span>Section</span><strong>{result.section}</strong></div><p>{result.count} linked passages found</p></div>
              {result.entries.length === 0 ? <EmptyState icon={History} title="No linked versions found" text="Try another section number or inspect the Corpus Library." /> : (
                <div className="timeline-list">
                  {result.entries.map((entry, index) => <TimelineEntry entry={entry} index={index} key={entry.chunk_id} />)}
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </PageFrame>
  )
}

function CorpusLibrary() {
  const [documents, setDocuments] = useState([])
  const [query, setQuery] = useState('')
  const [type, setType] = useState('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.documents().then(setDocuments).finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => documents.filter((doc) => {
    const matchesQuery = `${doc.title} ${doc.year}`.toLowerCase().includes(query.toLowerCase())
    return matchesQuery && (type === 'all' || doc.document_type === type)
  }), [documents, query, type])

  const types = ['all', ...new Set(documents.map((doc) => doc.document_type))]

  return (
    <PageFrame eyebrow="Official corpus" title="Browse the indexed GST collection" description="Inspect coverage, document status, effective dates, extraction counts, and official sources.">
      <div className="library-toolbar">
        <div className="search-input"><Search size={17} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search title or year" /></div>
        <div className="type-tabs">{types.map((item) => <button key={item} className={type === item ? 'active' : ''} onClick={() => setType(item)}>{item === 'all' ? 'All' : item.replaceAll('_', ' ')}</button>)}</div>
      </div>
      {loading ? <LoadingState compact /> : (
        <div className="document-grid">
          {filtered.map((doc) => <DocumentCard document={doc} key={doc.document_id} />)}
        </div>
      )}
    </PageFrame>
  )
}

function PageFrame({ eyebrow, title, description, children }) {
  return (
    <div className="page-wrap section-wrap">
      <div className="page-heading"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>
      {children}
    </div>
  )
}

function EvidenceList({ evidence }) {
  const [open, setOpen] = useState(0)
  return (
    <div className="evidence-section">
      <div className="subsection-title"><div><span className="eyebrow">Evidence inspector</span><h3>Retrieved passages</h3></div><span>{evidence?.length || 0} results</span></div>
      <div className="evidence-list">
        {evidence?.map((item, index) => (
          <article className={cx('evidence-card', open === index && 'expanded')} key={item.chunk_id}>
            <button className="evidence-summary" onClick={() => setOpen(open === index ? -1 : index)}>
              <span className="evidence-rank">{String(index + 1).padStart(2, '0')}</span>
              <span className="evidence-title"><strong>{item.title}</strong><small>Section {item.section_number} · Page {item.page_number}</small></span>
              <span className="score-chip">RRF {item.scores.rrf?.toFixed(3) || '—'}</span>
              <ChevronDown size={18} />
            </button>
            {open === index && (
              <div className="evidence-body">
                <p>{item.text}</p>
                <div className="score-row"><Score label="Semantic" value={item.scores.semantic} /><Score label="BM25" value={item.scores.bm25} /><Score label="Fused RRF" value={item.scores.rrf} /></div>
                <div className="metadata-row"><span><CalendarDays size={14} /> {formatRange(item.version_start || item.effective_from, item.version_end || item.effective_to)}</span><span><Layers3 size={14} /> {item.version_status?.replaceAll('_', ' ')}</span></div>
                <a href={item.source_url} target="_blank" rel="noreferrer">Open official PDF <ExternalLink size={14} /></a>
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  )
}

function CompactEvidence({ item, index }) {
  return (
    <div className="compact-evidence">
      <span>{index + 1}</span>
      <div><strong>{item.title}</strong><small>§ {item.section_number} · p. {item.page_number}</small><p>{item.text.slice(0, 145)}{item.text.length > 145 ? '…' : ''}</p></div>
    </div>
  )
}

function TimelineEntry({ entry, index }) {
  const [open, setOpen] = useState(index < 2)
  return (
    <article className="timeline-entry">
      <div className="timeline-node"><span>{index + 1}</span></div>
      <div className="timeline-content">
        <button onClick={() => setOpen(!open)}>
          <div><span>{formatDate(entry.version_start || entry.effective_from) || 'Date not recorded'}</span><strong>{entry.title}</strong><small>Section {entry.section_number} · {entry.version_status?.replaceAll('_', ' ')}</small></div>
          <ChevronDown size={18} />
        </button>
        {open && <div className="timeline-detail"><p>{entry.text}</p><a href={entry.source_url} target="_blank" rel="noreferrer">Official source <ExternalLink size={13} /></a></div>}
      </div>
    </article>
  )
}

function DocumentCard({ document }) {
  return (
    <article className="document-card">
      <div className="document-top"><span className={cx('doc-type', document.document_type)}>{document.document_type?.replaceAll('_', ' ')}</span><span className="doc-year">{document.year}</span></div>
      <BookOpen size={25} />
      <h3>{document.title}</h3>
      <p>{document.authority}</p>
      <div className="doc-metadata"><span><Layers3 size={14} /> {document.chunk_count} chunks</span><span><CalendarDays size={14} /> {formatDate(document.effective_from) || 'Review date'}</span></div>
      <div className="document-footer"><span className="status-dot"><span /> {document.official_status?.replaceAll('_', ' ')}</span><a href={document.source_url} target="_blank" rel="noreferrer" aria-label={`Open ${document.title}`}><ExternalLink size={16} /></a></div>
    </article>
  )
}

function SystemBadge({ system }) {
  const data = SYSTEMS[system] || SYSTEMS.version_aware
  return <span className={cx('system-badge', data.tone)}>{data.short}</span>
}

function Score({ label, value }) {
  const percent = Math.max(0, Math.min(100, (Number(value) || 0) * (label === 'BM25' ? 5 : 100)))
  return <div className="score"><span>{label}<strong>{value == null ? '—' : Number(value).toFixed(3)}</strong></span><i><b style={{ width: `${percent}%` }} /></i></div>
}

function LoadingState({ compact = false }) {
  return <div className={cx('loading-state', compact && 'compact')}><span className="loader-orbit"><i /><b /></span><div><strong>Working through the evidence</strong><p>Retrieving passages and checking version metadata…</p></div></div>
}

function EmptyState({ icon: Icon, title, text }) {
  return <div className="empty-state"><span><Icon size={28} /></span><strong>{title}</strong><p>{text}</p></div>
}

function ErrorState({ message }) {
  return <div className="error-state"><CircleAlert size={20} /><div><strong>Could not complete the request</strong><p>{message}. Make sure the FastAPI server is running on port 8000.</p></div></div>
}

function Footer({ health }) {
  return (
    <footer className="site-footer section-wrap">
      <div className="footer-brand"><img src="/gst-lens-logo.png" alt="" /><div><strong>GST Lens</strong><span>Version-aware RAG for Bhutan GST</span></div></div>
      <p>Research prototype · Not legal advice · Verify every answer against the linked official source.</p>
      <span className="footer-status"><i /> {health ? 'Index ready' : 'Backend not connected'}</span>
    </footer>
  )
}

function formatDate(value) {
  if (!value) return ''
  const parsed = new Date(`${value}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).format(parsed)
}

function formatRange(start, end) {
  if (!start && !end) return 'Effective period under review'
  return `${formatDate(start) || 'Unknown'} — ${end ? formatDate(end) : 'Current / open'}`
}

export default App
