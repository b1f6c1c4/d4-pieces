import './App.css';
import PiecesSelector from './PiecesSelector';
import Board from './Board';
import Solution from './Solution';
import {useEffect, useState} from 'react';

const moduleLoader = window.Pieces({ locateFile: () => 'pieces.wasm' });

function App() {
  const [module, setModule] = useState();
  const [pieces, setPieces] = useState([]);
  const [board, setBoard] = useState(undefined);
  const [solution, setSolution] = useState(undefined);
  const [solutionId, setSolutionId] = useState(0);

  useEffect(() => {
    moduleLoader.then((module) => {
      window.Pieces = module;
      setModule(module);
      const pieces = [];
      for (let n = 1; n <= 8; n++) {
        const m = module.shape_count(n);
        for (let i = 0; i < m; i++) {
          const p = new module.Piece(module.shape_at(n, i));
          p.count = 0;
          pieces.push(p);
        }
      }
      setPieces(pieces)
      setBoard(new module.Shape(window.BigInt('0b111011111110111111101111111011111110011111100111111')));
    });
  }, []);

  const handleToggle = (func) => () => {
    const p = pieces.find(func);
    if (p.count)
      pieces.filter(func).forEach(p => p.count = 0);
    else
      pieces.filter(func).forEach(p => p.count = 1);
    setPieces([...pieces]);
  };

  return (
    <div className="App">
      <div>
        {board && (<Board shape={board} onToggle={(row, col) => {
          if (board.test(row, col))
            setBoard(board.clear(row, col));
          else
            setBoard(board.set(row, col));
        }} />)}
        {solution && solution.size() && (
          <Solution pieces={pieces} board={board} solution={solution.get(solutionId)} />
        )}
      </div>
      <div>
        <button onClick={() => {
          setSolution(undefined);
          setTimeout(() => {
            const vec = new module.VPiece();
            pieces.forEach(p => p.count && vec.push_back(p));
            setSolution(module.solve(vec, board));
            setSolutionId(0);
          }, 100);
        }}>Solve!</button>
        {solution && (
          <span>{solution.size()} Solutions found!</span>
        )}
        {solution && solution.size() && (
          <input
            type="number"
            max={solution.size() - 1}
            min={0}
            value={solutionId}
            onChange={(e) => setSolutionId(+e.target.value)}
            style={{ width: '50px', textAlign: 'center' }}
          />
        )}
      </div>
      <div className="button-row">
        <button onClick={handleToggle((p) => p.shape.size === 1)}>1</button>
        <button onClick={handleToggle((p) => p.shape.size === 2)}>2</button>
        <button onClick={handleToggle((p) => p.shape.size === 3)}>3</button>
        <button onClick={handleToggle((p) => p.shape.size === 4)}>4</button>
        <button onClick={handleToggle((p) => p.shape.size === 5)}>5</button>
        <button onClick={handleToggle((p) => p.shape.size === 6)}>6</button>
        <button onClick={handleToggle((p) => p.shape.size === 7)}>7</button>
        <button onClick={handleToggle((p) => p.shape.size === 8)}>8</button>
      </div>
      <PiecesSelector
        module={module}
        pieces={pieces}
        onChange={() => setPieces([...pieces])}
      />
    </div>
  );
}

export default App;
