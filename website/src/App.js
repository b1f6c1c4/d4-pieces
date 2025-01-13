import logo from './logo.svg';
import './App.css';
import PiecesSelector from './PiecesSelector';
import Board from './Board';
import Solution from './Solution';
import {useEffect, useReducer, useState} from 'react';

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
      setPieces([
        0b1111000010000000,
        0b1110000000110000,
        0b011000000100000011000000,
        0b100000001000000011100000,
        0b1110000011100000,
        0b1110000011000000,
        0b1110000010100000,
        0b1111000000100000,
      ].map((sh) => new module.Piece(new module.Shape(sh).normalize())));
      setBoard(new module.Shape(window.BigInt('0b11111100111111001111111011111110111111101111111011100000')));
    });
  }, []);

  return (
    <div className="App">
      {board && (<Board shape={board} onToggle={(row, col) => {
        if (board.test(row, col))
          setBoard(board.clear(row, col));
        else
          setBoard(board.set(row, col));
      }} />)}
      <button onClick={() => {
        setSolution(undefined);
        setTimeout(() => {
          setSolution(module.solve(board));
          setSolutionId(0);
        }, 100);
      }}>Solve!</button>
      {solution && (
        <p>{solution.size()} Solutions found!</p>
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
      {solution && solution.size() && (
        <Solution board={board} solution={solution.get(solutionId)} />
      )}
      <PiecesSelector
        module={module}
        pieces={pieces}
        onChange={() => setPieces([...pieces])}
      />
    </div>
  );
}

export default App;
