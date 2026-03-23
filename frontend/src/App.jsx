import { useState } from 'react';
import './index.css';

function App() {
  const [song, setSong] = useState('');
  const [count, setCount] = useState(5);
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!song.trim()) return;

    setLoading(true);
    setError(null);
    setRecommendations(null);

    try {
      // We will point to localhost:8000 where our FastAPI runs
      const res = await fetch(`http://localhost:8000/api/recommend?song=${encodeURIComponent(song)}&count=${count}`);
      
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to fetch recommendations');
      }

      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Vibe Matcher</h1>
        <p className="subtitle">Discover new tracks with AI</p>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <label htmlFor="song">Seed Song</label>
          <input 
            id="song"
            type="text" 
            placeholder="e.g. Teenage Dream" 
            value={song}
            onChange={(e) => setSong(e.target.value)}
            required
            autoComplete="off"
          />
        </div>

        <div className="input-group">
          <label htmlFor="count">Number of Recommendations</label>
          <input 
            id="count"
            type="number" 
            min="1" 
            max="20" 
            value={count}
            onChange={(e) => setCount(e.target.value)}
            required
          />
        </div>

        <button type="submit" disabled={loading || !song.trim()}>
          {loading ? (
            <><span className="loader"></span> Analyzing Vibes...</>
          ) : (
            'Get Recommendations'
          )}
        </button>
      </form>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {recommendations && recommendations.length > 0 && (
        <div className="results-container">
          <h2>Your Playlist</h2>
          <div className="track-list">
            {recommendations.map((track, idx) => (
              <div 
                key={idx} 
                className="track-card"
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                <div className="track-number">{idx + 1}</div>
                <div className="track-info">
                  <div className="track-name">{track.track_name}</div>
                  <div className="artist-name">{track.artist_names}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
