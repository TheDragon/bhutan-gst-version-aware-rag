import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight,
  ArrowUp,
  Lightbulb,
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
  standard: { label: 'Standard RAG', short: 'Standard', tone: 'slate', description: 'Searches for relevant wording across all dates.' },
  date_filtered: { label: 'Date-filtered RAG', short: 'Date-filtered', tone: 'blue', description: 'Searches only passages active on your chosen date.' },
  version_aware: { label: 'Version-aware RAG', short: 'Version-aware', tone: 'gold', description: 'Uses your date and amendment links to rank passages.' },
}

const today = new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Thimphu', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date())

function cx(...values) {
  return values.filter(Boolean).join(' ')
}

function App() {
  const [page, setPage] = useState('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const [stats, setStats] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    api.stats().then(setStats).catch(() => {})
    api.health().then(setHealth).catch(() => {})
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
        {page === 'research' && <ResearchWorkspace llm={health?.llm_model} />}
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
  const benchmark = stats?.systems?.version_aware
  const accuracy = benchmark?.version_selection_accuracy
  const questionCount = benchmark?.version_sensitive_questions
  const hasAccuracy = Number.isFinite(accuracy) && questionCount > 0
  const statCards = [
    { value: stats?.documents ?? '—', label: 'Official documents', icon: FileCheck2 },
    { value: stats?.chunks?.toLocaleString() ?? '—', label: 'Searchable passages', icon: Layers3 },
    { value: stats?.version_events ?? '—', label: 'Version relationships', icon: History },
    {
      value: hasAccuracy ? `${(accuracy * 100).toFixed(1)}%` : '—',
      label: 'Version-selection accuracy',
      detail: hasAccuracy ? `${Math.round(accuracy * questionCount)} of ${questionCount} version-sensitive test questions` : 'Benchmark result unavailable',
      icon: Gauge,
    },
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
            {health?.llm_model
              ? <span><Check size={14} /> {generatorLabel(health.llm_model)} answers</span>
              : health?.generator_model && <span><Check size={14} /> {generatorLabel(health.generator_model)}</span>}
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
            <div><strong>Amendment Act 2025</strong><small>Clause 12 · PDF page 24</small></div>
            <FileCheck2 size={20} />
          </div>
          <div className="research-card card-version">
            <History size={18} />
            <div><strong>Applicable version</strong><small>Effective 01 Jan 2026</small></div>
          </div>
        </div>
      </section>

      <section className="stats-band section-wrap" aria-label="Project statistics">
        {statCards.map(({ value, label, detail, icon: Icon }) => (
          <article className="stat-card" key={label}>
            <Icon size={19} />
            <strong>{value}</strong>
            <span>{label}{detail && <small>{detail}</small>}</span>
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

function ResearchWorkspace({ llm }) {
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const endRef = useRef(null)
  const busy = turns.some((turn) => turn.loading)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [turns.length])

  // The chat always asks about the law in force today with version-aware
  // retrieval. Questions about the past ("what was it before?", "in 2025")
  // are read from the wording; the Compare tab keeps the date and mode controls.
  const ask = async (question) => {
    const text = question.trim()
    if (text.length < 3 || busy) return
    const id = Date.now()
    setDraft('')
    setTurns((current) => [...current, { id, question: text, loading: true }])
    try {
      const result = await api.answer({ question: text, system: 'version_aware', top_k: 5 })
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, loading: false, result } : turn)))
    } catch (err) {
      setTurns((current) => current.map((turn) => (turn.id === id ? { ...turn, loading: false, error: err.message } : turn)))
    }
  }

  return (
    <div className="page-wrap section-wrap chat-page">
      <div className="chat-heading">
        <span className="eyebrow">Ask GST Lens</span>
        <h1>Ask about Bhutan’s GST law</h1>
        <p>Answers use the law in force today, quote the official text, and show when a rule was amended. Ask “what was it before?” to see an earlier version.</p>
      </div>

      <div className="chat-thread">
        {turns.length === 0 && (
          <div className="chat-empty">
            <strong>Try asking</strong>
            <div className="starter-grid">
              {STARTERS.map((question) => (
                <button key={question} onClick={() => ask(question)} disabled={busy}>
                  <span>{question}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        {turns.map((turn) => (
          <section className="chat-turn" key={turn.id}>
            <div className="user-bubble"><p>{turn.question}</p></div>
            {turn.loading && <LoadingState compact />}
            {turn.error && <ErrorState message={turn.error} />}
            {turn.result && <AnswerTurn result={turn.result} onAsk={ask} busy={busy} />}
          </section>
        ))}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={(event) => { event.preventDefault(); ask(draft) }}>
        <textarea
          rows={2}
          value={draft}
          placeholder="Ask about a GST rule, rate, product or section…"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              ask(draft)
            }
          }}
        />
        <button className="send-button" disabled={busy || draft.trim().length < 3} aria-label="Send question">
          {busy ? <span className="spinner" /> : <ArrowUp size={18} />}
        </button>
      </form>
      <p className="composer-note">
        Research prototype, not legal advice.{' '}
        {llm ? `Answers are written by ${generatorLabel(llm)} from the official texts only.` : 'Offline mode: answers quote the law directly.'}{' '}
        To compare retrieval methods or pick a date, use Compare systems.
      </p>
    </div>
  )
}

const STARTERS = [
  'What is the GST rate?',
  'What was the GST rate before the latest change?',
  'Who reviews a GST decision if I disagree with the Department?',
  'Do I pay GST on cooking oil?',
]

function AnswerTurn({ result, onAsk, busy }) {
  if (result.clarification_title) {
    return (
      <article className="answer-card">
        <p className="answer-text">{result.answer}</p>
        {result.clarification_question && (
          <button className="secondary-button clarification-button" disabled={busy} onClick={() => onAsk(result.clarification_question)}>
            {result.clarification_action} <ArrowRight size={15} />
          </button>
        )}
      </article>
    )
  }

  const citation = result.citations?.[0]
  const selected = result.version_context?.selected
  const compare = result.version_context?.compare
  const quoted = isQuotedPassage(result)
  // Without an answer model the "answer" is itself a passage of the law; show
  // it as the quote rather than pretending it is a written answer.
  const quote = quoted ? result.answer : result.evidence_quote

  return (
    <article className="answer-card">
      {result.time_note && <p className="time-note"><History size={14} /> {result.time_note}</p>}
      <p className={cx('answer-text', quoted && 'muted')}>
        {quoted && !result.abstained ? 'This is the most relevant wording I found in the law:' : result.answer}
      </p>

      {!result.abstained && quote && (
        <figure className="answer-source-quote">
          <figcaption>From the law</figcaption>
          <blockquote>“{quote}”</blockquote>
          {citation && <SourceLine citation={citation} selected={selected} />}
        </figure>
      )}
      {!result.abstained && !quote && citation && <div className="source-only"><SourceLine citation={citation} selected={selected} /></div>}

      {!result.abstained && compare && <AmendmentBlock compare={compare} summary={result.change_summary} />}

      {result.suggestion && (
        <div className="suggestion">
          <Lightbulb size={15} />
          <span>{result.abstained ? 'Try asking:' : 'This answer may be incomplete. For a fuller answer, try:'}</span>
          <button disabled={busy} onClick={() => onAsk(result.suggestion)}>{result.suggestion} <ArrowRight size={13} /></button>
        </div>
      )}
    </article>
  )
}

function SourceLine({ citation, selected }) {
  const name = selected?.short_name || citation.title
  return (
    <p className="source-line">
      {name} · {sectionLabel(selected || citation)} · page {citation.page_number}
      {citation.source_url && <> · <a href={citation.source_url} target="_blank" rel="noreferrer">Open official PDF <ExternalLink size={12} /></a></>}
    </p>
  )
}

function AmendmentBlock({ compare, summary }) {
  const earlier = compare.direction === 'earlier'
  const { before, after } = compare
  const used = <em className="used-tag">used for this answer</em>
  return (
    <div className="amendment">
      <p className="amendment-head"><History size={15} /> {earlier ? 'This rule was amended' : 'This rule was amended later'}</p>
      <ol className="amendment-timeline">
        <li className={cx(before.selected && 'used')}>
          <b>{before.short_name}</b>
          <small>{formatDate(before.start)} – {formatDate(before.end) || 'later'}</small>
          {before.selected && used}
        </li>
        <li className="arrow" aria-hidden="true"><ArrowRight size={16} /></li>
        <li className={cx(after.selected && 'used')}>
          <b>{after.short_name}</b>
          <small>in force from {formatDate(after.start)}</small>
          {after.selected && used}
        </li>
      </ol>
      {summary && <p className="amendment-summary">{summary}</p>}
      {compare.diff?.length > 0 ? (
        <p className="diff-text">
          {compare.diff.map((op, index) => {
            if (op.op === 'del') return <del key={index}>{op.text}</del>
            if (op.op === 'add') return <ins key={index}>{op.text}</ins>
            return <span key={index}>{op.text}</span>
          }).reduce((parts, part, index) => (index ? [...parts, ' ', part] : [part]), [])}
        </p>
      ) : (
        <div className="side-by-side">
          <div><span>Before</span><p>{before.text}</p></div>
          <div><span>After</span><p>{after.text}</p></div>
        </div>
      )}
      {compare.amended_by && <p className="amended-by">Changed by {compare.amended_by}</p>}
    </div>
  )
}

function sectionLabel(item) {
  const section = String(item.section_number || '')
  if (!section || section.startsWith('page-part')) return 'Schedule / notice text'
  const amendment = item.document_type === 'amendment' || /amendment_/.test(item.document_id || '')
  if (amendment) {
    const target = item.affected_sections?.[0]
    return target ? `Clause ${section} (amends Section ${target})` : `Clause ${section}`
  }
  return `${item.document_id?.includes('rules') ? 'Rule' : 'Section'} ${section}`
}

function CompareWorkspace() {
  const [question, setQuestion] = useState('How was GST Act Section 40 amended as of 1 January 2026?')
  const [targetDate, setTargetDate] = useState('2026-01-01')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)
    try {
      setResult(await api.compare({ question, target_date: targetDate || null, top_k: 3, generate_answers: false }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageFrame eyebrow="Compare systems" title="What does each search method find?" description="Ask one question and pick a date. You’ll see which passages each method puts first.">
      <form className="compare-form" onSubmit={run}>
        <div><label htmlFor="compare-question">Your GST question</label><input id="compare-question" value={question} onChange={(e) => setQuestion(e.target.value)} /></div>
        <div><label htmlFor="compare-date">Target date</label><input id="compare-date" type="date" value={targetDate} onChange={(e) => setTargetDate(e.target.value)} /></div>
        <button className="primary-button" disabled={loading || question.trim().length < 3}>{loading ? <span className="spinner" /> : <BarChart3 size={17} />} Compare</button>
      </form>
      {error && <ErrorState message={error} />}
      {loading && <LoadingState compact />}
      {result && (
        <section className="comparison-guide" aria-live="polite">
          <h2>{comparisonSummary(result.comparisons)}</h2>
          <p>Start with <strong>Rank 1</strong>, each method’s first pick. Open any result to read the full passage, see which sections it changes, and check the official source.</p>
        </section>
      )}
      <div className="comparison-grid">
        {Object.entries(SYSTEMS).map(([system, method]) => {
          const evidence = result?.comparisons.find((column) => column.system === system)?.evidence
          return (
            <article className={cx('comparison-column', system)} key={system}>
              <header className="comparison-head">
                <h2>{method.short}</h2>
                <p>{method.description}</p>
              </header>
              {evidence?.length === 0 && <p className="comparison-empty">No passages returned for this search.</p>}
              {evidence?.map((item, index) => <CompactEvidence item={item} index={index} key={item.chunk_id} />)}
            </article>
          )
        })}
      </div>
    </PageFrame>
  )
}

function comparisonSummary(comparisons) {
  const first = comparisons.map(({ evidence }) => evidence[0]?.chunk_id)
  if (first.some((id) => !id)) return 'Some methods returned no passages'
  if (new Set(first).size === 1) return 'All three methods put the same passage first'

  const firstBySystem = Object.fromEntries(comparisons.map(({ system }, index) => [system, first[index]]))
  if (firstBySystem.standard === firstBySystem.date_filtered) {
    return 'Standard and Date-filtered agree. Version-aware puts a different passage first.'
  }
  return 'The methods differ on which passage comes first'
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

function CompactEvidence({ item, index }) {
  const locatorLabel = { amendment: 'Clause', rules: 'Rule' }[item.document_type] ?? 'Section'
  const locator = item.section_number && !item.section_number.startsWith('page-part-')
    ? `${locatorLabel} ${item.section_number}`
    : ''
  const reference = [locator, item.page_number && `PDF page ${item.page_number}`].filter(Boolean).join(' · ')
  const excerpt = item.text.length > 175 ? `${item.text.slice(0, 175).replace(/\s+\S*$/, '')}…` : item.text

  return (
    <details className="compact-evidence">
      <summary>
        <span className={cx('comparison-rank', index === 0 && 'first')}>Rank {index + 1}</span>
        <strong>{item.title}</strong>
        {reference && <small>{reference}</small>}
        <span className="comparison-excerpt">{excerpt}</span>
        <span className="comparison-disclosure">
          <span className="expand-label">Read passage</span><span className="collapse-label">Hide passage</span><ChevronDown size={15} />
        </span>
      </summary>
      <div className="comparison-passage">
        <p>{item.text}</p>
        {item.affected_sections?.length > 0 && <p><strong>Affected sections</strong> {item.affected_sections.join(', ')}</p>}
        {item.source_url && <a href={item.source_url} target="_blank" rel="noreferrer">Open official source <ExternalLink size={14} /></a>}
      </div>
    </details>
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

function LoadingState({ compact = false }) {
  return <div className={cx('loading-state', compact && 'compact')}><span className="loader-orbit"><i /><b /></span><div><strong>Working through the evidence</strong><p>Retrieving passages and checking version metadata…</p></div></div>
}

function EmptyState({ icon: Icon, title, text }) {
  return <div className="empty-state"><span><Icon size={28} /></span><strong>{title}</strong><p>{text}</p></div>
}

function ErrorState({ message }) {
  return <div className="error-state"><CircleAlert size={20} /><div><strong>Could not complete the request</strong><p>{message}</p></div></div>
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

function isQuotedPassage(result) {
  return ['extractive', 'validated_extractive_fallback'].includes(result?.answer_strategy)
}

function generatorLabel(model) {
  return model === 'extractive-evidence-v1' ? 'Extractive answers' : model.replace(/^google\//, '')
}

function formatRange(start, end) {
  if (!start && !end) return 'Effective period under review'
  return `${formatDate(start) || 'Unknown'} — ${end ? formatDate(end) : 'Current / open'}`
}

export default App
