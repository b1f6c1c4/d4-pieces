import logo from './logo.svg';
import './App.css';
import PiecesSelector from './PiecesSelector';
import Board from './Board';
import Solution from './Solution';
import {useEffect, useReducer, useState} from 'react';

const moduleLoader = window.Pieces({ locateFile: () => 'pieces.wasm' });

function App() {
  const [lib, setLib] = useState(undefined);
  const [board, setBoard] = useState(undefined);
  const [solution, setSolution] = useState(undefined);
  const [solutionId, setSolutionId] = useState(0);
  const [, lolUpdateAnyway] = useReducer(x => x + 1, 0);

  useEffect(() => {
    moduleLoader.then((module) => {
      window.Pieces = module;
      const lib = new module.Library();
      [
        0b1111000010000000,
        0b1110000000110000,
        0b011000000100000011000000,
        0b100000001000000011100000,
        0b1110000011100000,
        0b1110000011000000,
        0b1110000010100000,
        0b1111000000100000,
      ].forEach((sh) => lib.push(new module.Shape(sh)));
      setLib(lib);
      setBoard(new module.Shape(window.BigInt('0b11111100111111001111111011111110111111101111111011100000')));
    });
  }, []);

  const shapes = [];
  if (lib) {
    for (let i = 0; i < lib.length; i++) {
      shapes.push(lib.at(i));
    }
  }

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
          setSolution(lib.solve(board));
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
      <PiecesSelector shapes={shapes} onAvailChange={(id, value) => {
        shapes[id].count = value;
        lolUpdateAnyway();
      }} />
    </div>
  );
}

export default App;
