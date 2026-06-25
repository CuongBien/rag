import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import ForceGraph2D from 'react-force-graph-2d';
import { Bot, TestTube, Send, ArrowRightLeft, AlertCircle } from 'lucide-react';
import './App.css';

function App() {
  // Chat State
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Chào bạn! Tôi là trợ lý ảo Hóa học. Tôi đã được nạp sẵn Dữ liệu Đồ thị (Graph) và Không gian Vector. Bạn muốn hỏi gì?' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Translate State
  const [translateQuery, setTranslateQuery] = useState('');
  const [translateDirection, setTranslateDirection] = useState('iupac_to_smiles');
  const [translateResult, setTranslateResult] = useState(null);
  const [isTranslateLoading, setIsTranslateLoading] = useState(false);

  // Tabs
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'translate', 'sandbox', 'poly'

  // Polypharmacy State
  const [polyPrescription, setPolyPrescription] = useState('Aspirin, Clopidogrel, Omeprazole, Celecoxib, Lithium');
  const [polyGraphData, setPolyGraphData] = useState({ nodes: [], links: [] });
  const [polyAnalysis, setPolyAnalysis] = useState('');
  const [isPolyLoading, setIsPolyLoading] = useState(false);

  // Sandbox State
  const [availableDrugs, setAvailableDrugs] = useState([]);
  const [sandboxSearchTerm, setSandboxSearchTerm] = useState('');
  const [sandboxNodes, setSandboxNodes] = useState([]);
  const [sandboxGraphData, setSandboxGraphData] = useState({ nodes: [], links: [] });
  const [isSandboxLoading, setIsSandboxLoading] = useState(false);

  useEffect(() => {
    axios.get('/api/drugs').then(res => setAvailableDrugs(res.data.drugs)).catch(console.error);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = { role: 'user', content: chatInput };
    setMessages(prev => [...prev, userMsg]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const res = await axios.post(`/api/chat`, { query: userMsg.content });
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: res.data.answer,
        graphData: res.data.graph_data 
      }]);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message;
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Lỗi: ${errorMsg}` }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleTranslate = async (e) => {
    e.preventDefault();
    if (!translateQuery.trim()) return;

    setIsTranslateLoading(true);
    setTranslateResult(null);

    try {
      const res = await axios.post(`/api/translate`, {
        query: translateQuery,
        direction: translateDirection
      });
      setTranslateResult(res.data);
    } catch (error) {
      setTranslateResult({ error: `Lỗi kết nối: ${error.message}` });
    } finally {
      setIsTranslateLoading(false);
    }
  };

  const handleDragStart = (e, drug) => {
    e.dataTransfer.setData('drugId', drug);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    const drug = e.dataTransfer.getData('drugId');
    if (!drug || sandboxNodes.includes(drug)) return;
    
    const newNodes = [...sandboxNodes, drug];
    setSandboxNodes(newNodes);
    
    setIsSandboxLoading(true);
    try {
      const res = await axios.post('/api/sandbox/relations', { drugs: newNodes });
      setSandboxGraphData(res.data.graph_data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSandboxLoading(false);
    }
  };

  const removeSandboxNode = async (drug) => {
    const newNodes = sandboxNodes.filter(n => n !== drug);
    setSandboxNodes(newNodes);
    
    if (newNodes.length === 0) {
      setSandboxGraphData({ nodes: [], links: [] });
      return;
    }
    
    setIsSandboxLoading(true);
    try {
      const res = await axios.post('/api/sandbox/relations', { drugs: newNodes });
      setSandboxGraphData(res.data.graph_data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSandboxLoading(false);
    }
  };

  const handlePolypharmacy = async (e) => {
    e.preventDefault();
    if (!polyPrescription.trim()) return;
    
    setIsPolyLoading(true);
    setPolyAnalysis('');
    try {
      const res = await axios.post('/api/polypharmacy', { prescription: polyPrescription });
      setPolyGraphData(res.data.graph_data);
      setPolyAnalysis(res.data.analysis);
    } catch (err) {
      console.error(err);
      setPolyAnalysis('Lỗi kết nối tới máy chủ. Vui lòng thử lại.');
    } finally {
      setIsPolyLoading(false);
    }
  };

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo-icon">🏥</div>
          <div className="logo-text">
            <h2>MedGraph</h2>
            <span>AI Platform</span>
          </div>
        </div>
        <nav className="side-menu">
          <button className={`menu-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
            <Bot size={20} /> Knowledge Assistant
          </button>
          <button className={`menu-item ${activeTab === 'translate' ? 'active' : ''}`} onClick={() => setActiveTab('translate')}>
            <TestTube size={20} /> Chemical Translator
          </button>
          <button className={`menu-item ${activeTab === 'sandbox' ? 'active' : ''}`} onClick={() => setActiveTab('sandbox')}>
            <div style={{fontSize:'20px', width: '20px', display:'flex', justifyContent:'center'}}>🕸️</div> Graph Sandbox
          </button>
          <button className={`menu-item ${activeTab === 'poly' ? 'active' : ''}`} onClick={() => setActiveTab('poly')} style={{color: activeTab === 'poly' ? '#ef4444' : ''}}>
            <AlertCircle size={20} /> Polypharmacy
          </button>
        </nav>
      </aside>

      <main className="main-content light-theme">
        <header className="top-bar">
          <h2>
            {activeTab === 'chat' ? 'Knowledge Assistant' : 
             activeTab === 'translate' ? 'Chemical Translator' : 
             activeTab === 'sandbox' ? 'Graph Sandbox' : 'Polypharmacy Optimizer'}
          </h2>
        </header>

        <div className="content-scroll">
        {activeTab === 'chat' && (
        <section className="white-panel">
          
          <div className="chat-container">
            <div className="chat-history">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  {msg.graphData && (
                    <div className="graph-wrapper">
                      <ForceGraph2D
                        graphData={msg.graphData}
                        width={400}
                        height={300}
                        nodeRelSize={6}
                        nodeColor={() => '#10b981'}
                        nodeCanvasObjectMode={() => 'after'}
                        nodeCanvasObject={(node, ctx, globalScale) => {
                          const label = node.name;
                          const fontSize = 14 / globalScale;
                          ctx.font = `${fontSize}px Sans-Serif`;
                          ctx.textAlign = 'center';
                          ctx.textBaseline = 'middle';
                          ctx.fillStyle = '#1e293b';
                          ctx.fillText(label, node.x, node.y + 10);
                        }}
                        linkColor={() => '#94a3b8'}
                        linkCurvature="curvature"
                        linkCanvasObjectMode={() => 'after'}
                        linkCanvasObject={(link, ctx, globalScale) => {
                          const start = link.source;
                          const end = link.target;
                          if (typeof start !== 'object' || typeof end !== 'object') return;
                          
                          const dx = end.x - start.x;
                          const dy = end.y - start.y;
                          const c = link.curvature || 0;
                          const textPos = { x: start.x + dx / 2 + dy * c, y: start.y + dy / 2 - dx * c };
                          
                          const fontSize = 12 / globalScale;
                          ctx.font = `${fontSize}px Sans-Serif`;
                          ctx.fillStyle = '#64748b';
                          ctx.textAlign = 'center';
                          ctx.textBaseline = 'middle';
                          ctx.fillText(link.label, textPos.x, textPos.y);
                        }}
                        linkDirectionalArrowLength={0}
                        backgroundColor="#f8fafc"
                        onNodeDragEnd={node => {
                          node.fx = node.x;
                          node.fy = node.y;
                        }}
                      />
                    </div>
                  )}
                </div>
              ))}
              {isChatLoading && (
                <div className="message assistant">
                  <div className="loading-spinner"></div> Suy nghĩ...
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form className="chat-input-area" onSubmit={handleChat}>
              <input
                type="text"
                className="chat-input"
                placeholder="Hỏi về Aspirin, Caffeine..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={isChatLoading}
              />
              <button type="submit" disabled={isChatLoading || !chatInput.trim()}>
                <Send size={18} />
              </button>
            </form>
          </div>
        </section>
        )}

        {activeTab === 'translate' && (
        <section className="white-panel">

          <form className="translator-form" onSubmit={handleTranslate}>
            <select 
              value={translateDirection}
              onChange={(e) => setTranslateDirection(e.target.value)}
            >
              <option value="iupac_to_smiles">IUPAC to SMILES</option>
              <option value="smiles_to_iupac">SMILES to IUPAC</option>
            </select>

            <input
              type="text"
              placeholder={translateDirection === 'iupac_to_smiles' ? 'Nhập tên IUPAC (VD: Aspirin)' : 'Nhập chuỗi SMILES'}
              value={translateQuery}
              onChange={(e) => setTranslateQuery(e.target.value)}
            />

            <button type="submit" disabled={isTranslateLoading || !translateQuery.trim()}>
              {isTranslateLoading ? <div className="loading-spinner"></div> : (
                <>
                  <ArrowRightLeft size={18} />
                  Dịch thuật
                </>
              )}
            </button>
          </form>

          {translateResult && !translateResult.error && (
            <div className={`translation-result ${translateResult.prediction === 'None' ? 'error' : ''}`}>
              <div className="result-label">Kết quả ({translateDirection === 'iupac_to_smiles' ? 'SMILES' : 'IUPAC'})</div>
              <div className="result-value">
                {translateResult.prediction === 'None' ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444' }}>
                    <AlertCircle size={20} /> Hóa chất vô lý (Zero-Hallucination)
                  </span>
                ) : (
                  <>
                    <div>{translateResult.prediction}</div>
                    {translateDirection === 'iupac_to_smiles' && translateResult.prediction && !translateResult.error && (
                      <img 
                        src={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(translateResult.prediction)}/PNG`} 
                        alt="Molecule" 
                        className="molecule-img"
                      />
                    )}
                  </>
                )}
              </div>
              <div className="result-source">Source: {translateResult.source}</div>
            </div>
          )}

          {translateResult?.error && (
            <div className="translation-result error">
              <AlertCircle size={20} /> {translateResult.error}
            </div>
          )}
        </section>
        )}

        {activeTab === 'sandbox' && (
          <div className="sandbox-layout">
            <aside className="sandbox-sidebar white-panel">
              <div className="panel-title">
                <span>Drugs Database ({availableDrugs.length})</span>
              </div>
              <div className="sandbox-search">
                <input 
                  type="text" 
                  placeholder="Tìm kiếm thuốc..." 
                  value={sandboxSearchTerm}
                  onChange={e => setSandboxSearchTerm(e.target.value)}
                />
              </div>
              <div className="drug-list">
                {availableDrugs
                  .filter(d => d.toLowerCase().includes(sandboxSearchTerm.toLowerCase()))
                  .map(drug => (
                  <div 
                    key={drug} 
                    className={`drug-card ${sandboxNodes.includes(drug) ? 'in-canvas' : ''}`}
                    draggable={!sandboxNodes.includes(drug)}
                    onDragStart={(e) => handleDragStart(e, drug)}
                  >
                    {drug}
                  </div>
                ))}
              </div>
            </aside>
            <section className="sandbox-canvas white-panel" style={{padding: 0, overflow: 'hidden'}} onDragOver={handleDragOver} onDrop={handleDrop}>
              {sandboxNodes.length === 0 ? (
                <div className="empty-canvas-msg">Kéo thả thuốc từ danh sách bên trái vào đây...</div>
              ) : (
                <div className="canvas-wrapper">
                  <div className="canvas-toolbar">
                    <button onClick={() => { setSandboxNodes([]); setSandboxGraphData({nodes:[],links:[]}); }} className="clear-btn">Xóa tất cả</button>
                  </div>
                  {isSandboxLoading && <div className="sandbox-loader"><div className="loading-spinner"></div> Loading relations...</div>}
                  <ForceGraph2D
                    graphData={sandboxGraphData}
                    width={950}
                    height={750}
                    nodeRelSize={8}
                    nodeColor={(node) => `hsl(${(node.group * 50) % 360}, 70%, 60%)`}
                    nodeCanvasObjectMode={() => 'replace'}
                    nodeCanvasObject={(node, ctx, globalScale) => {
                      const label = node.name;
                      const fontSize = 14 / globalScale;
                      
                      // Draw circle
                      ctx.beginPath();
                      ctx.arc(node.x, node.y, 10, 0, 2 * Math.PI, false);
                      ctx.fillStyle = `hsl(${(node.group * 50) % 360}, 70%, 60%)`;
                      ctx.fill();
                      ctx.strokeStyle = '#ffffff';
                      ctx.lineWidth = 1.5 / globalScale;
                      ctx.stroke();

                      // Draw text
                      ctx.font = `${fontSize}px Sans-Serif`;
                      ctx.textAlign = 'center';
                      ctx.textBaseline = 'middle';
                      ctx.fillStyle = '#1e293b';
                      ctx.fillText(label, node.x, node.y + 18);
                    }}
                    linkColor={() => '#94a3b8'}
                    linkCurvature="curvature"
                    linkCanvasObjectMode={() => 'after'}
                    linkCanvasObject={(link, ctx, globalScale) => {
                      const start = link.source;
                      const end = link.target;
                      if (typeof start !== 'object' || typeof end !== 'object') return;
                      
                      const dx = end.x - start.x;
                      const dy = end.y - start.y;
                      const c = link.curvature || 0;
                      const textPos = { x: start.x + dx / 2 + dy * c, y: start.y + dy / 2 - dx * c };
                      
                      const fontSize = 12 / globalScale;
                      ctx.font = `${fontSize}px Sans-Serif`;
                      ctx.fillStyle = '#64748b';
                      ctx.textAlign = 'center';
                      ctx.textBaseline = 'middle';
                      ctx.fillText(link.label, textPos.x, textPos.y);
                    }}
                    linkDirectionalArrowLength={0}
                    backgroundColor="#f8fafc"
                    onNodeDragEnd={node => {
                      node.fx = node.x;
                      node.fy = node.y;
                    }}
                  />
                  <div className="active-nodes-list">
                    {sandboxNodes.map(n => (
                      <span key={n} className="active-node-badge">
                        {n} <button onClick={() => removeSandboxNode(n)}>×</button>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>
        )}

        {activeTab === 'poly' && (
          <div className="polypharmacy-layout">
            <section className="white-panel" style={{marginBottom: '1.5rem'}}>
              <div className="panel-title">
                <span style={{color: '#ef4444', fontWeight: 'bold'}}>🚨 Polypharmacy Optimizer (Kiểm soát Đa thuốc)</span>
              </div>
              <p style={{marginBottom: '1rem', color: '#94a3b8'}}>Nhập danh sách các thuốc trong toa (cách nhau bằng dấu phẩy) để hệ thống GraphDB + Llama3 dò tìm các tương tác nguy hiểm ẩn sâu bên trong mạng lưới y khoa.</p>
              <form onSubmit={handlePolypharmacy} style={{display: 'flex', gap: '1rem'}}>
                <input 
                  type="text" 
                  value={polyPrescription} 
                  onChange={e => setPolyPrescription(e.target.value)}
                  placeholder="Ví dụ: Aspirin, Clopidogrel, Omeprazole, Celecoxib..."
                  style={{flex: 1, padding: '1rem', fontSize: '1.1rem'}}
                />
                <button type="submit" disabled={isPolyLoading} style={{background: '#ef4444'}}>
                  {isPolyLoading ? <div className="loading-spinner"></div> : 'Phân tích Toa thuốc'}
                </button>
              </form>
            </section>

            <div className="poly-results" style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', height: '600px'}}>
              <section className="white-panel poly-graph" style={{padding: 0, overflow: 'hidden'}}>
                <div style={{flex: 1, position: 'relative', background: '#f8fafc', height: '100%'}}>
                  {polyGraphData.nodes.length > 0 ? (
                    <ForceGraph2D
                      graphData={polyGraphData}
                      width={700}
                      height={500}
                      nodeRelSize={8}
                      nodeColor={() => '#3b82f6'}
                      nodeCanvasObjectMode={() => 'replace'}
                      nodeCanvasObject={(node, ctx, globalScale) => {
                        const fontSize = 14 / globalScale;
                        ctx.beginPath(); ctx.arc(node.x, node.y, 10, 0, 2*Math.PI, false);
                        ctx.fillStyle = '#3b82f6'; ctx.fill();
                        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5/globalScale; ctx.stroke();
                        ctx.font = `${fontSize}px Sans-Serif`; ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle'; ctx.fillStyle = '#1e293b';
                        ctx.fillText(node.name, node.x, node.y + 18);
                      }}
                      linkColor={() => '#ef4444'}
                      linkWidth={2}
                      linkCurvature="curvature"
                      linkCanvasObjectMode={() => 'after'}
                      linkCanvasObject={(link, ctx, globalScale) => {
                        const start = link.source; const end = link.target;
                        if (typeof start !== 'object' || typeof end !== 'object') return;
                        
                        const dx = end.x - start.x;
                        const dy = end.y - start.y;
                        const c = link.curvature || 0;
                        const textPos = { x: start.x + dx / 2 + dy * c, y: start.y + dy / 2 - dx * c };
                        
                        ctx.font = `${12/globalScale}px Sans-Serif`; ctx.fillStyle = '#ef4444';
                        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                        ctx.fillText(link.label, textPos.x, textPos.y);
                      }}
                      linkDirectionalArrowLength={0}
                      onNodeDragEnd={node => { node.fx = node.x; node.fy = node.y; }}
                    />
                  ) : (
                    <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8'}}>
                      Chưa có dữ liệu phân tích. Hãy nhập toa thuốc.
                    </div>
                  )}
                </div>
              </section>

              <section className="white-panel poly-report" style={{overflowY: 'auto'}}>
                <h3 style={{marginTop: 0, color: '#0ea5e9'}}>Báo cáo An toàn Lâm sàng</h3>
                <div className="report-content" style={{lineHeight: '1.6', fontSize: '1.05rem', color: '#1e293b'}}>
                  {isPolyLoading ? (
                    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem'}}>
                      <div className="loading-spinner" style={{width: '3rem', height: '3rem', borderColor: '#ef4444', borderTopColor: 'transparent'}}></div>
                      <span style={{color: '#ef4444', fontWeight: 'bold'}}>Đang quét GraphDB & Llama-3 phân tích...</span>
                    </div>
                  ) : polyAnalysis ? (
                    <ReactMarkdown>{polyAnalysis}</ReactMarkdown>
                  ) : (
                    <span style={{color: '#94a3b8'}}>AI Dược sĩ đang chờ xử lý toa thuốc...</span>
                  )}
                </div>
              </section>
            </div>
          </div>
        )}
        </div>
      </main>
    </div>
  );
}

export default App;
