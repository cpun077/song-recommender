import { useState } from 'react';
import './index.css';

function App() {
  const [song, setSong] = useState('');
  const [count, setCount] = useState(5);
  const [recommendations, setRecommendations] = useState(null);
  const [searchResults, setSearchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!song.trim()) return;

    setLoading(true);
    setError(null);
    setRecommendations(null);
    setSearchResults(null);

    try {
      const res = await fetch(`http://localhost:8000/api/search?query=${encodeURIComponent(song)}`);
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to search songs');
      }

      if (data.results.length === 0) {
        setError("No songs found matching your query.");
        setLoading(false);
      } else if (data.results.length === 1) {
        handleRecommend(data.results[0].track_id);
      } else {
        setSearchResults(data.results);
        setLoading(false);
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleRecommend = async (trackId) => {
    setLoading(true);
    setError(null);
    setRecommendations(null);
    setSearchResults(null);

    try {
      const res = await fetch(`http://localhost:8000/api/recommend?song=${encodeURIComponent(trackId)}&count=${count}`);
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
        <h1>Songs Like</h1>
        <p className="subtitle">Discover music that actually sounds similar</p>
      </header>

      <form onSubmit={handleSearch}>
        <div className="input-group">
          <label htmlFor="song">Song</label>
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
          <label htmlFor="count">Number of Results</label>
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
            'Search & Recommend'
          )}
        </button>
      </form>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {searchResults && searchResults.length > 1 && (
        <div className="results-container">
          <h2>Select a Song</h2>
          <div className="track-list">
            {searchResults.map((track, idx) => (
              <div
                key={idx}
                className="track-card"
                style={{ animationDelay: `${idx * 0.05}s`, cursor: 'pointer' }}
                onClick={() => handleRecommend(track.track_id)}
              >
                <div className="track-info">
                  <div className="track-name">{track.track_name}</div>
                  <div className="artist-name">{track.artist_names?.replace(/\|/g, ', ')}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {recommendations && recommendations.length > 0 && (
        <div className="results-container">
          <h2>Your Similar Playlist</h2>
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
                  <div className="artist-name">{track.artist_names?.replace(/\|/g, ', ')}</div>
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
